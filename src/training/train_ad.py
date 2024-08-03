import datetime
import gc
import inspect
import logging
import math
import os
import random
import subprocess
import sys
import time
from enum import Enum
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler as get_lr_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

import training
import wandb
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.animation import AnimationPipeline
from animatediff.schedulers import get_scheduler
from animatediff.utils.device import get_memory_format, get_model_dtypes
from animatediff.utils.util import (relative_path, save_frames, save_images,
                                    save_video)
from training.dataset_ad import make_dataloader

from .utils import LogType, zero_rank_partial

logger = logging.getLogger(__name__)
zero_rank_print: Callable[[str, LogType], None] = partial(zero_rank_partial, logger)

def train_ad(
    image_finetune: bool,

    name: str,
    use_wandb: bool,

    output_dir: str,
    sd_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    device_id: int,

    epoch_size:int = 1000,
    num_epochs:int = 1,

    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,

    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,

    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 1,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    # Initialize distributed training
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    sample_start_time = time.time()
    sample_end_time = time.time()

    # Logging folder
    run_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    folder_name = "debug" if is_debug else run_name
    run_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(run_dir):
        os.system(f"rm -rf {run_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(f"{run_dir}/samples", exist_ok=True)
        os.makedirs(f"{run_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(run_dir, 'config.yaml'))

    # set up scheduler
    noise_scheduler = get_scheduler("ddim", OmegaConf.to_container(noise_scheduler_kwargs))
    zero_rank_print(f'Using scheduler "ddim" ({noise_scheduler.__class__.__name__})', LogType.info)
    logger.debug("Loading tokenizer...")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")
    logger.debug("Loading text encoder...")
    text_encoder: CLIPSkipTextModel = CLIPSkipTextModel.from_pretrained(sd_model_path, subfolder="text_encoder")
    logger.debug("Loading VAE...")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(sd_model_path, subfolder="vae")
    logger.debug("Loading Unet...")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            sd_model_path, subfolder="unet",
            motion_module_path="data/models/motion-module/mm_sd_v15_v2.safetensors",
            # motion_module_path="data/models/motion-module/mm_sd_v15_v2.ckpt",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(sd_model_path, subfolder="unet")

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"Loading from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        raw_state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        # Modify the keys by removing 'module.' prefix if it exists
        state_dict = {k.replace('module.', ''): v for k, v in raw_state_dict.items()}

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                # zero_rank_print(f"Training module: {name}", LogType.debug)
                param.requires_grad = True
                break

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    zero_rank_print(f"trainable params number: {len(trainable_params)}")
    zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # enable xformers if available
    if enable_xformers_memory_efficient_attention:
        logger.info("Enabling xformers memory-efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    # Enable gradient checkpointing
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        unet.enable_gradient_checkpointing()

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Move models to GPU

    # unet_dtype, tenc_dtype, vae_dtype = get_model_dtypes(device_id, force_half)
    # model_memory_format = get_memory_format(device_id)

    # text_encoder = text_encoder.to(device=device_id, dtype=tenc_dtype)
    # vae = vae.to(device=device_id, dtype=vae_dtype, memory_format=model_memory_format)

    text_encoder = text_encoder.to(device=device_id)
    vae = vae.to(device=device_id)

    # Get the training dataset
    train_dataloader = make_dataloader(**train_data, batch_size=train_batch_size, num_workers=num_workers)

    # Get the training iteration
    # if max_train_steps == -1:
    #     assert max_train_epoch != -1
    #     max_train_steps = max_train_epoch * len(train_dataset)

    # if checkpointing_steps == -1:
    #     assert checkpointing_epochs != -1
    #     checkpointing_steps = checkpointing_epochs * len(train_dataset)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_lr_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=epoch_size * num_epochs * gradient_accumulation_steps / train_batch_size,
    )

    # DDP wrapper
    unet_single = unet.to(device=device_id)
    # unet_single = torch.compile(unet_single)
    unet = DDP(unet_single, device_ids=[device_id], output_device=device_id)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {epoch_size}")
        logging.info(f"  Num Epochs = {num_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {epoch_size * num_epochs * gradient_accumulation_steps / train_batch_size}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, num_epochs*epoch_size), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.amp.GradScaler('cuda') if mixed_precision_training else None

    def normalize_and_rescale(image):
        MEAN = [0.5, 0.5, 0.5]
        SD = [0.5, 0.5, 0.5]
        mean = torch.tensor(MEAN).view(1, 1, 3, 1, 1).to(image.device)
        std = torch.tensor(SD).view(1, 1, 3, 1, 1).to(image.device)
        return (image/255.0 - mean) / std

    unet.train()
    for epoch in range(first_epoch, num_epochs):
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            sample_start_time = time.time()
            data_wait_time = sample_start_time - sample_end_time

            pixel_values = batch[0].to(device_id)
            texts = batch[1]

            pixel_values = normalize_and_rescale(pixel_values)
            # zero_rank_print(f"Pixel shape {pixel_values}")
            # pixel_values = rearrange(pixel_values, "b f h w c -> b f c h w")

            if cfg_random_null_text:
                texts = [name if random.random() > cfg_random_null_text_ratio else "" for name in texts]

            # Data batch sanity check
            if epoch == first_epoch and step < 4:
                sanity_pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                if not image_finetune:
                    for idx, (pixel_value, text) in enumerate(zip(sanity_pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        # save_frames(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}")
                        save_video(pixel_value.cpu(), f"{run_dir}/sanity_check/{step}-{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{step}-{idx}'}.mp4")
                else:
                    for idx, (pixel_value, text) in enumerate(zip(sanity_pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value.cpu(), f"{run_dir}/sanity_check/{step}-{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{step}-{idx}'}.png")

            ### >>>> Training >>>> ###

            # Periodically validation
            actual_steps = global_step/gradient_accumulation_steps
            if is_main_process and actual_steps in validation_steps_tuple or actual_steps % validation_steps == 0:
                zero_rank_print("Validation")

                # Validation pipeline
                if not image_finetune:
                    validation_pipeline = AnimationPipeline(
                        unet=unet_single, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, feature_extractor=None,
                    ).to(device_id)
                else:
                    validation_pipeline = StableDiffusionPipeline.from_pretrained(
                        sd_model_path,
                        unet=unet_single, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
                    )
                validation_pipeline.enable_vae_slicing()

                generator = torch.Generator(device=vae.device)
                generator.manual_seed(global_seed)

                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                # prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                for idx, prompt in enumerate(validation_data.prompts):
                    if not image_finetune:
                        with torch.inference_mode(True):
                            pipeline_output = validation_pipeline(
                                prompt,
                                generator    = generator,
                                video_length = train_data.sample_n_frames,
                                height       = height,
                                width        = width,
                                num_inference_steps = validation_data.get("num_inference_steps", 25),
                                guidance_scale      = validation_data.get("guidance_scale", 7.5),
                                context_frames = train_data.sample_n_frames,
                                context_stride = 1,
                                context_overlap = 4,
                            ).videos
                        save_video(pipeline_output, f"{run_dir}/samples/sample-{actual_steps}/{idx}.mp4")
                    else:
                        with torch.inference_mode(True):
                            pipeline_output = validation_pipeline(
                                prompt,
                                generator           = generator,
                                height              = height,
                                width               = width,
                                num_inference_steps = validation_data.get("num_inference_steps", 25),
                                guidance_scale      = validation_data.get("guidance_scale", 8.),
                            ).images[0]
                        pipeline_output = torchvision.transforms.functional.to_tensor(pipeline_output)
                        save_images([pipeline_output], f"{run_dir}/samples/sample-{actual_steps}/")

                logging.info(f"Saved samples to {run_dir}")
                del validation_pipeline, pipeline_output
                torch.cuda.empty_cache()
                gc.collect()

            zero_rank_print("Convert videos to latent space", LogType.debug)
            # Convert videos to latent space
            # pixel_values = batch["pixel_values"].to(device_id, dtype=vae_dtype, memory_format=model_memory_format)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    # pixel_values = pixel_values.to(device_id, dtype=vae_dtype, memory_format=model_memory_format)
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    # pixel_values = pixel_values.to(device_id, dtype=vae_dtype, memory_format=model_memory_format)
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            zero_rank_print("Sample noise", LogType.debug)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz, _, num_frames, _, _ = latents.shape

            zero_rank_print(f"noise {noise.shape}", LogType.debug)
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device_id)
            timesteps = timesteps.long()

            zero_rank_print(f"timesteps {timesteps.shape}", LogType.debug)
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            zero_rank_print(f"noisy_latents {noisy_latents.shape}", LogType.debug)

            zero_rank_print("Get text embedding", LogType.debug)
            # Get the text embedding for conditioning
            zero_rank_print(f"Texts {texts}")
            with torch.no_grad():
                prompt_ids = tokenizer(
                    texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(device_id)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            del latents, batch
            torch.cuda.empty_cache()

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Reset gradients at the beginning of the accumulation cycle
            if step % gradient_accumulation_steps == 0:
                zero_rank_print(f"Reset gradients at the beginning of the accumulation cycle", LogType.debug)
                optimizer.zero_grad()

            # Predict the noise residual and compute loss
            # Mixed-precision training
            zero_rank_print(f"Predict the noise residual and compute loss Mixed Precision: {mixed_precision_training}", LogType.debug)
            with torch.amp.autocast('cuda', enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / gradient_accumulation_steps  # Normalize loss to account for accumulation

            del noisy_latents, noise, timesteps, model_pred
            torch.cuda.empty_cache()

            # Backpropagate loss, accumulate gradients
            zero_rank_print("Backpropagate", LogType.debug)
            if mixed_precision_training:
                scaler.scale(loss).backward()  # Scale the loss; backward pass accumulates gradients
            else:
                loss.backward()  # Gradient accumulation without scaling

            # Apply the optimizer step and update the learning rate scheduler only at the end of an accumulation period
            if (step + 1) % gradient_accumulation_steps == 0:
                zero_rank_print("=== Accumulate gradients", LogType.debug)
                if mixed_precision_training:
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    scaler.step(optimizer)  # Perform optimizer step
                    scaler.update()  # Update the scale for next iteration
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    optimizer.step()

                lr_scheduler.step()  # Update learning rate

                sample_end_time = time.time()
                sample_time = sample_end_time - sample_start_time
                # Log to WandB
                if is_main_process and step > 1 and (not is_debug) and use_wandb:
                    wandb.log({
                        "train_loss": loss.item() * gradient_accumulation_steps,
                        "epoch": epoch,
                        "sample_time": sample_time,
                        "data_wait_time": data_wait_time
                    })
                    zero_rank_print(f"train_loss {loss.item() * gradient_accumulation_steps} epoch {epoch} sample_time {sample_time} data_wait_time {data_wait_time}", LogType.debug)
                epoch_loss += loss.item() * gradient_accumulation_steps

            # Update the progress bar
            progress_bar.set_postfix(loss=epoch_loss / (step + 1))
            global_step += 1
            ### <<<< Training <<<< ###

            del loss
            torch.cuda.empty_cache()

        # Save checkpoint
        if is_main_process:
            state_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "state_dict": unet.module.state_dict(),
            }
            torch.save(state_dict, f"{run_name}_animatediff_epoch_{epoch}.pth")
            wandb.save(f"animatediff_{run_name}_epoch_{epoch}.pth")
            logging.info(f"Saved state to checkpoints {run_name} (global_step: {global_step})")

        # Additional cleanup at the end of an epoch if applicable
        gc.collect()  # Force garbage collection

    dist.destroy_process_group()


