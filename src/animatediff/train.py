import datetime
import gc
import inspect
import logging
import math
import os
import random
import subprocess
import sys
from enum import Enum
from typing import Dict, Optional, Tuple

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
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

import wandb
from animatediff.dataset import make_dataloader, make_dataset
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.animation import AnimationPipeline
from animatediff.schedulers import get_scheduler
from animatediff.utils.device import get_memory_format, get_model_dtypes
from animatediff.utils.util import relative_path, save_frames, save_video

logger = logging.getLogger(__name__)

class LogType(str, Enum):
    info = "info"
    debug = "debug"
    error = "error"

def zero_rank_print(s, logtype:LogType = LogType.info):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
        if logtype == LogType.info:
            logger.info(s)
        elif logtype == LogType.debug:
            logger.debug(s)
        elif logtype == LogType.error:
            logger.error(s)

def init_dist(launcher="slurm", backend='gloo', port=29500, **kwargs):
    """Initializes distributed environment."""
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['WORLD_SIZE'] = '1'
    # os.environ['RANK'] = '0'
    # os.environ['MASTER_PORT'] = str(port)
    try:
        if launcher == 'pytorch':
            rank = int(os.environ['RANK'])
            num_gpus = torch.cuda.device_count()
            local_rank = rank % num_gpus
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend, **kwargs)

        elif launcher == 'slurm':
            proc_id = int(os.environ['SLURM_PROCID'])
            ntasks = int(os.environ['SLURM_NTASKS'])
            node_list = os.environ['SLURM_NODELIST']
            num_gpus = torch.cuda.device_count()
            local_rank = proc_id % num_gpus
            torch.cuda.set_device(local_rank)
            addr = subprocess.getoutput(
                f'scontrol show hostname {node_list} | head -n1')
            os.environ['MASTER_ADDR'] = addr
            os.environ['WORLD_SIZE'] = str(ntasks)
            os.environ['RANK'] = str(proc_id)
            port = os.environ.get('PORT', port)
            os.environ['MASTER_PORT'] = str(port)
            dist.init_process_group(backend=backend)
            zero_rank_print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        else:
            raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    except KeyError:
        logging.error("Environment variables not set. Use 'torchrun --nnodes=1 --nproc_per_node=1' to launch.")
        sys.exit(1)
    return local_rank


def train_ad(
    image_finetune: bool,

    name: str,
    use_wandb: bool,
    launcher: str,
    use_xformers: bool,
    force_half: bool,

    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,

    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,

    max_train_epoch: int = -1,
    max_train_steps: int = 100,
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
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # set up scheduler
    noise_scheduler = get_scheduler("ddim", OmegaConf.to_container(noise_scheduler_kwargs))
    zero_rank_print(f'Using scheduler "ddim" ({noise_scheduler.__class__.__name__})', LogType.info)
    logger.debug("Loading tokenizer...")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    logger.debug("Loading text encoder...")
    text_encoder: CLIPSkipTextModel = CLIPSkipTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    logger.debug("Loading VAE...")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    logger.debug("Loading Unet...")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet",
            motion_module_path="C:/dev/animatediff-cli/data/models/motion-module/mm_sd_v15_v2.ckpt",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"Loading from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

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
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    zero_rank_print(f"trainable params number: {len(trainable_params)}")
    zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # enable xformers if available
    if use_xformers:
        logger.info("Enabling xformers memory-efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    # Enable gradient checkpointing
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        unet.enable_gradient_checkpointing()

    # Move models to GPU

    # unet_dtype, tenc_dtype, vae_dtype = get_model_dtypes(local_rank, force_half)
    # model_memory_format = get_memory_format(local_rank)

    # text_encoder = text_encoder.to(device=local_rank, dtype=tenc_dtype)
    # vae = vae.to(device=local_rank, dtype=vae_dtype, memory_format=model_memory_format)

    text_encoder = text_encoder.to(device=local_rank)
    vae = vae.to(device=local_rank)

    # Get the training dataset
    train_dataset = make_dataset(**train_data)
    train_dataloader = make_dataloader(train_dataset, batch_size=train_batch_size)

    # # Get the training dataset
    # train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    # distributed_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=num_processes,
    #     rank=global_rank,
    #     shuffle=True,
    #     seed=global_seed,
    # )

    # # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=train_batch_size,
    #     shuffle=False,
    #     sampler=distributed_sampler,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataset)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataset)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_lr_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # DDP wrapper
    unet_single = unet.to(device=local_rank)
    # unet_single = torch.compile(unet_single)
    unet = DDP(unet_single, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {num_train_epochs}")
        logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:

                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        # save_frames(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}")
                        save_video(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif")
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")

            # Periodically validation
            if is_main_process and (global_step in validation_steps_tuple):
                zero_rank_print("Validation")

                # Validation pipeline
                if not image_finetune:
                    validation_pipeline = AnimationPipeline(
                        unet=unet_single, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, feature_extractor=None,
                    ).to(local_rank)
                else:
                    validation_pipeline = StableDiffusionPipeline.from_pretrained(
                        pretrained_model_path,
                        unet=unet_single, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
                    )
                validation_pipeline.enable_vae_slicing()

                samples = []

                generator = torch.Generator(device=vae.device)
                generator.manual_seed(global_seed)

                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                # prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                for idx, prompt in enumerate(validation_data.prompts):
                    if not image_finetune:
                        with torch.inference_mode(True):
                            sample = validation_pipeline(
                                prompt,
                                generator    = generator,
                                video_length = train_data.sample_n_frames,
                                height       = height,
                                width        = width,
                                **validation_data,
                            ).videos
                        save_video(sample, f"{output_dir}/samples/sample-{global_step}/{idx}.gif")
                        samples.append(sample)
                    else:
                        with torch.inference_mode(True):
                            sample = validation_pipeline(
                                prompt,
                                generator           = generator,
                                height              = height,
                                width               = width,
                                num_inference_steps = validation_data.get("num_inference_steps", 25),
                                guidance_scale      = validation_data.get("guidance_scale", 8.),
                            ).images[0]
                        sample = torchvision.transforms.functional.to_tensor(sample)
                        samples.append(sample)

                logging.info(f"Saved samples to {output_dir}")
                del validation_pipeline, samples, sample
                torch.cuda.empty_cache()
                gc.collect()

            ### >>>> Training >>>> ###

            zero_rank_print("Convert videos to latent space", LogType.debug)
            # Convert videos to latent space
            # pixel_values = batch["pixel_values"].to(local_rank, dtype=vae_dtype, memory_format=model_memory_format)
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    # pixel_values = pixel_values.to(local_rank, dtype=vae_dtype, memory_format=model_memory_format)
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    # pixel_values = pixel_values.to(local_rank, dtype=vae_dtype, memory_format=model_memory_format)
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            zero_rank_print("Sample noise", LogType.debug)
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=local_rank)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            zero_rank_print("Get text embedding", LogType.debug)
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(local_rank)
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
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
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

            # Apply the optimizer step and update the learning rate scheduler only at the end of an accumulation period or at the last batch
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataset) - 1:
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

            progress_bar.update(1)
            global_step += 1
            ### <<<< Training <<<< ###

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataset) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                }
                if step == len(train_dataset) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Logging
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Wandb logging
            if is_main_process and (not is_debug) and use_wandb:
                wandb.log({"train_loss": (loss * gradient_accumulation_steps).item()}, step=global_step)

            del loss
            torch.cuda.empty_cache()

            if global_step >= max_train_steps:
                break

        # Additional cleanup at the end of an epoch if applicable
        gc.collect()  # Force garbage collection

    dist.destroy_process_group()


