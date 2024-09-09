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
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler as get_lr_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from lion_pytorch import Lion
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
from animatediff.utils.isolate_rng import isolate_rng
from animatediff.utils.util import (relative_path, save_frames, save_images,
                                    save_video)
from training.dataset_ad import make_dataloader

from .utils import LogType, apply_lora, zero_rank_partial

logger = logging.getLogger(__name__)
zero_rank_print: Callable[[str, LogType], None] = partial(zero_rank_partial, logger)

# actual prediction function - shared between train and validate
def get_model_prediction_and_target(batch, unet, vae, noise_scheduler, tokenizer, text_encoder, generator, run_dir, cfg_random_null_text_ratio=0.0, zero_frequency_noise_ratio=0.0, return_loss=False, loss_scale=None, embedding_perturbation=0.0, mixed_precision_training: bool = True, image_finetune: bool = False, fps:int = 6, sanity_check: bool = False):
    with torch.no_grad():
        with torch.autocast('cuda', enabled=mixed_precision_training):
            pixel_values = batch[0].to(unet.device)
            texts = batch[1]

            if cfg_random_null_text_ratio > 0:
                texts = [name if torch.rand(1, generator=generator).item() > cfg_random_null_text_ratio else "" for name in texts]

            batch_size = pixel_values.shape[0]
            video_length = pixel_values.shape[1]
            pixel_values = normalize_and_rescale(pixel_values, image_finetune=image_finetune)

            # if sanity_check:
            #     sanity_pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
            #     if not image_finetune:
            #         for idx, (pixel_value, text) in enumerate(zip(sanity_pixel_values, texts)):
            #             pixel_value = pixel_value[None, ...]
            #             save_video(pixel_value.cpu(), f"{run_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{idx}'}.mp4", fps=fps)
            #     else:
            #         for idx, (pixel_value, text) in enumerate(zip(sanity_pixel_values, texts)):
            #             pixel_value = pixel_value / 2. + 0.5
            #             torchvision.utils.save_image(pixel_value.cpu(), f"{run_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{idx}'}.png")

            zero_rank_print("Convert videos to latent space", LogType.debug)
            if not image_finetune:
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
            else:
                latents = vae.encode(pixel_values).latent_dist
                latents = latents.sample()
            latents = latents * 0.18215

        del pixel_values, batch

        zero_rank_print("Get text embedding", LogType.debug)
        zero_rank_print(f"Texts {texts}")
        prompt_ids = tokenizer(
            texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(unet.device)
        encoder_hidden_states = text_encoder(prompt_ids, output_hidden_states=True).last_hidden_state
        del texts, prompt_ids

        zero_rank_print("Sample noise", LogType.debug)
        if image_finetune:
            noise = torch.randn(latents.size(), generator=generator).to(device=latents.device, dtype=latents.dtype)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), generator=generator).to(device=unet.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            lookahead_denoising = False
        else:
            noise, noisy_latents, timesteps, lookahead_denoising = add_noise_same_timestep(noise_scheduler, latents, batch_size, video_length, generator=generator, device=unet.device)
        zero_rank_print(f"Lookahead {lookahead_denoising}")
        zero_rank_print(f"noise {noise.shape}", LogType.debug) #[2, 4, 16, 32, 32]
        zero_rank_print(f"latents {latents.shape}", LogType.debug)
        zero_rank_print(f"timesteps {timesteps.shape}", LogType.debug) # 32
        zero_rank_print(f"noisy_latents {noisy_latents.shape}", LogType.debug)


    # https://arxiv.org/pdf/2405.20494
    # perturbation_deviation = embedding_perturbation / math.sqrt(encoder_hidden_states.shape[2])
    # perturbation_delta =  torch.randn_like(encoder_hidden_states) * (perturbation_deviation)
    # encoder_hidden_states = encoder_hidden_states + perturbation_delta

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type in ["v_prediction", "v-prediction"]:
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        del noise, latents

    with torch.autocast('cuda', enabled=mixed_precision_training):
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        zero_rank_print(f"model_pred {model_pred.shape}", LogType.debug)

    if return_loss:
        if loss_scale is None:
            loss_scale = torch.ones(model_pred.shape[0], dtype=torch.float)

        # if args.min_snr_gamma is not None:
        #     snr = compute_snr(timesteps, noise_scheduler)

        #     mse_loss_weights = (
        #             torch.stack(
        #                 [snr, args.min_snr_gamma * torch.ones_like(timesteps)], dim=1
        #             ).min(dim=1)[0]
        #             / snr
        #     )
        #     mse_loss_weights[snr == 0] = 1.0
        #     loss_scale = loss_scale * mse_loss_weights.to(loss_scale.device)
        if lookahead_denoising:
            mid_point = model_pred.shape[2] // 2
            loss_mse = F.mse_loss(model_pred[:, :, mid_point:, :, :].float(), target[:, :, mid_point:, :, :].float(), reduction="none")
        else:
            loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")

        if image_finetune:
            loss_scale = loss_scale.view(-1, 1, 1, 1).expand_as(loss_mse)
        else:
            loss_scale = loss_scale.view(-1, 1, 1, 1, 1).expand_as(loss_mse)

        # if args.loss_type == "mse_huber":
        #     early_timestep_bias = (timesteps / noise_scheduler.config.num_train_timesteps)
        #     early_timestep_bias = torch.tensor(early_timestep_bias, dtype=torch.float).to(unet.device)
        #     early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(loss_mse)
        #     loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction="none", delta=1.0)
        #     loss_mse = loss_mse * loss_scale.to(unet.device) * early_timestep_bias
        #     loss_huber = loss_huber * loss_scale.to(unet.device) * (1.0 - early_timestep_bias)
        #     loss = loss_mse.mean() + loss_huber.mean()
        # elif args.loss_type == "huber_mse":
        #     early_timestep_bias = (timesteps / noise_scheduler.config.num_train_timesteps)
        #     early_timestep_bias = torch.tensor(early_timestep_bias, dtype=torch.float).to(unet.device)
        #     early_timestep_bias = early_timestep_bias.view(-1, 1, 1, 1).expand_as(loss_mse)
        #     loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction="none", delta=1.0)
        #     loss_mse = loss_mse * loss_scale.to(unet.device) * (1.0 - early_timestep_bias)
        #     loss_huber = loss_huber * loss_scale.to(unet.device) * early_timestep_bias
        #     loss = loss_huber.mean() + loss_mse.mean()
        # elif args.loss_type == "huber":
        #     loss_huber = F.huber_loss(model_pred.float(), target.float(), reduction="none", delta=1.0)
        #     loss_huber = loss_huber * loss_scale.to(unet.device)
        #     loss = loss_huber.mean()
        # else:
        loss_mse = loss_mse * loss_scale.to(unet.device)
        loss = loss_mse.mean()

        return model_pred, target, loss
    else:
        return model_pred, target

def normalize_and_rescale(image, image_finetune=False):
    MEAN = [0.5, 0.5, 0.5]
    SD = [0.5, 0.5, 0.5]
    # MEAN = [0.485, 0.456, 0.406]
    # SD = [0.229, 0.224, 0.225]
    if image_finetune:
        mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor(SD).view(1, 3, 1, 1).to(image.device)
    else:
        mean = torch.tensor(MEAN).view(1, 1, 3, 1, 1).to(image.device)
        std = torch.tensor(SD).view(1, 1, 3, 1, 1).to(image.device)
    return (image.float()/255.0 - mean) / std

def add_noise_same_timestep(noise_scheduler, latents, batch_size, video_length, generator, device):
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    # Sample noise that we'll add to the latents
    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return noise, noisy_latents, timesteps, False

def add_noise_random_timesteps(noise_scheduler, latents, batch_size, video_length, generator, device):
    # Sample noise that we'll add to the latents
    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size * video_length,), device=device).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    noisy_latents = rearrange(noisy_latents, "(b f) c h w -> b c f h w", f=video_length)
    noise = rearrange(noise, "(b f) c h w -> b c f h w", f=video_length)
    timesteps = rearrange(timesteps, '(b f) -> b f', f=video_length)
    return noise, noisy_latents, timesteps, False

def add_noise_sequential_timesteps(noise_scheduler, latents, batch_size, video_length, generator, device):
    target_partitions = 2
    target_steps = video_length * target_partitions #32

    # Sample noise that we'll add to the latents
    noise_scheduler.set_timesteps(target_steps)
    timesteps = noise_scheduler.timesteps
    # timesteps = torch.flip(timesteps, dims=[0])
    timesteps = torch.cat([torch.full((video_length//2,), timesteps[0]), timesteps])

    rank = random.randint(0, 2 * target_partitions - 1)
    start_idx = rank*(video_length // 2)
    end_idx = start_idx + video_length
    timesteps = timesteps[start_idx:end_idx].to(device).long()
    print(f"timesteps {timesteps}")

    noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    noisy_latents = rearrange(noisy_latents, "(b f) c h w -> b c f h w", f=video_length)
    noise = rearrange(noise, "(b f) c h w -> b c f h w", f=video_length)
    timesteps = rearrange(timesteps, '(b f) -> b f', f=video_length)

    print(f"Timesteps {timesteps}")

    return noise, noisy_latents, timesteps, True

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
    resume_id: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,

    validation_steps: int = 100,
    validation_gen_steps: int = 100,
    validation_gen_steps_tuple: Tuple = (-1,),

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

    global_step = 0
    first_epoch = 0

    seed = global_seed + global_rank

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
        if resume_id != "":
            run = wandb.init(project="animatediff", name=folder_name, config=config, id=resume_id, resume="must")
        else:
            run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(f"{run_dir}/samples", exist_ok=True)
        os.makedirs(f"{run_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(run_dir, 'config.yaml'))

    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
            # motion_module_path="data/models/motion-module/mm_sd_v15_v2.safetensors",
            motion_module_path="data/models/motion-module/mm_sd_v15_v2.ckpt",
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(sd_model_path, subfolder="unet")

    # Load pretrained unet weights
    if unet_checkpoint_path != "":
        zero_rank_print(f"Loading from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path:
            zero_rank_print(f"global_step: {unet_checkpoint_path['global_step']}")
            if resume_id != "":
                global_step = unet_checkpoint_path['global_step']
        raw_state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path

        # Modify the keys by removing 'module.' prefix if it exists
        state_dict = {k.replace('module.', ''): v for k, v in raw_state_dict.items()}

        m, u = unet.load_state_dict(state_dict, strict=False)
        zero_rank_print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if train_data.adapter_lora_path != "":
        print(f"load domain lora from {train_data.adapter_lora_path}")
        domain_lora_state_dict = torch.load(train_data.adapter_lora_path, map_location="cpu")
        domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
        domain_lora_state_dict.pop("animatediff_config", "")
        unet = apply_lora(unet, domain_lora_state_dict, alpha=train_data.adapter_lora_scale)

    weight_dtype = torch.float32
    if mixed_precision_training:
        weight_dtype = torch.float16

    # Freeze vae and text_encoder
    vae.requires_grad_(False).to(weight_dtype)
    text_encoder.requires_grad_(False).to(weight_dtype)

    # Set unet trainable parameters
    unet.requires_grad_(False).to(weight_dtype)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                # zero_rank_print(f"Training module: {name}", LogType.debug)
                param.requires_grad = True
                break

    if mixed_precision_training:
        cast_training_params(unet, dtype=torch.float32)

    torch.backends.cuda.matmul.allow_tf32 = True

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

    if not train_data.use_lion_optim:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )
    else:
        optimizer = Lion(
            trainable_params,
            lr=learning_rate / 10,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay * 10,
        )

    # Move models to GPU
    text_encoder = text_encoder.to(device=device_id)
    vae = vae.to(device=device_id)

    # Get the training dataset
    train_dataloader = make_dataloader(**train_data, shardshuffle=100, resampled=(resume_id != ""), batch_size=train_batch_size, num_workers=num_workers, epoch_size=epoch_size*train_batch_size*gradient_accumulation_steps, seed=seed, is_image=image_finetune)

    val_dataloader = make_dataloader(**validation_data, shardshuffle=False, batch_size=1, num_workers=0, epoch_size=validation_data.val_size, is_image=image_finetune)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_lr_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=epoch_size * num_epochs,
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

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, num_epochs*epoch_size*total_batch_size), disable=not is_main_process)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.amp.GradScaler('cuda') if mixed_precision_training else None

    loss_epoch = []
    loss_log_step = []
    loss_mean = 0

    unet.train()
    for epoch in range(first_epoch, num_epochs):
        for step, batch in enumerate(train_dataloader):

            data_wait_time = time.time() - sample_end_time

            ### >>>> Training >>>> ###

            # Periodically validation
            actual_steps = global_step/gradient_accumulation_steps

            if is_main_process and actual_steps % validation_steps == 0:
            # and actual_steps > 0:
                with torch.no_grad(), isolate_rng():

                    torch.manual_seed(validation_data.get("seed"))
                    np.random.seed(validation_data.get("seed"))
                    random.seed(validation_data.get("seed"))

                    val_generator = torch.Generator(device="cpu")
                    val_generator.manual_seed(validation_data.get("seed"))

                    loss_validation_epoch = []
                    steps_pbar = tqdm(range(validation_data.val_size), position=1, leave=False)
                    steps_pbar.set_description(f"Validation")

                    for val_step, val_batch in enumerate(val_dataloader):
                        model_pred, target = get_model_prediction_and_target(val_batch, unet, vae, noise_scheduler, tokenizer, text_encoder, val_generator, run_dir, return_loss=False, mixed_precision_training=mixed_precision_training, image_finetune=image_finetune, fps=train_data.target_fps, sanity_check=val_step==0)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        del target, model_pred

                        loss_step = loss.detach().item()
                        loss_validation_epoch.append(loss_step)

                        steps_pbar.update(1)

                    steps_pbar.close()

                    loss_validation_local = sum(loss_validation_epoch) / len(loss_validation_epoch)
                    wandb.log({
                        "val_loss": loss_validation_local,
                    }, step=int(actual_steps))

            if is_main_process and (actual_steps in validation_gen_steps_tuple or actual_steps % validation_gen_steps) == 0:
                # and actual_steps > 0
            # if False:
                zero_rank_print("Validation Gen")

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

                val_generator = torch.Generator(device="cpu")
                val_generator.manual_seed(validation_data.get("seed"))

                height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size

                # prompts = validation_data.prompts[:2] if global_step < 1000 and (not image_finetune) else validation_data.prompts

                for idx, prompt in enumerate(validation_data.prompts):
                    if not image_finetune:
                        with torch.inference_mode(True):
                            pipeline_output = validation_pipeline(
                                prompt,
                                generator    = val_generator,
                                video_length = train_data.sample_n_frames,
                                height       = height,
                                width        = width,
                                num_inference_steps = validation_data.get("num_inference_steps", 25),
                                guidance_scale      = validation_data.get("guidance_scale", 8),
                                context_frames = train_data.sample_n_frames,
                                context_stride = 1,
                                context_overlap = 4,
                            ).videos
                        save_video(pipeline_output, f"{run_dir}/samples/sample-{actual_steps}/{idx}.mp4", validation_data.get("fps", 8))
                    else:
                        with torch.inference_mode(True):
                            pipeline_output = validation_pipeline(
                                prompt,
                                generator           = val_generator,
                                height              = height,
                                width               = width,
                                num_inference_steps = validation_data.get("num_inference_steps", 25),
                                guidance_scale      = validation_data.get("guidance_scale", 8.),
                            ).images[0]
                        pipeline_output = torchvision.transforms.functional.to_tensor(pipeline_output)
                        save_images([pipeline_output], f"{run_dir}/samples/sample-{actual_steps}/{idx}.png")

                logging.info(f"Saved samples to {run_dir}")
                del validation_pipeline, pipeline_output

            torch.cuda.empty_cache()
            gc.collect()

            sample_start_time = time.time()

            model_pred, target, loss = get_model_prediction_and_target(batch, unet, vae, noise_scheduler, tokenizer, text_encoder, train_generator, run_dir, cfg_random_null_text_ratio=cfg_random_null_text_ratio, return_loss=True, mixed_precision_training=mixed_precision_training, image_finetune=image_finetune, fps=train_data.target_fps, sanity_check=global_step==0)

            del target, model_pred
            torch.cuda.empty_cache()

            # Backpropagate loss, accumulate gradients
            zero_rank_print("Backpropagate", LogType.debug)
            if mixed_precision_training:
                scaler.scale(loss).backward()  # Scale the loss; backward pass accumulates gradients
            else:
                loss.backward()  # Gradient accumulation without scaling

            loss_step = loss.detach().item()
            loss_log_step.append(loss_step)

            # Apply the optimizer step and update the learning rate scheduler only at the end of an accumulation period
            if (step + 1) % gradient_accumulation_steps == 0:
                zero_rank_print("=== Accumulate gradients", LogType.debug)
                if mixed_precision_training:
                    if train_data.use_clipping:
                        scaler.unscale_(optimizer)  # Unscale gradients before clipping
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    scaler.step(optimizer)  # Perform optimizer step
                    scaler.update()  # Update the scale for next iteration
                else:
                    if train_data.use_clipping:
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    optimizer.step()

                lr_scheduler.step()  # Update learning rate

                sample_end_time = time.time()
                sample_time = sample_end_time - sample_start_time
                # Log to WandB
                loss_mean = sum(loss_log_step) / len(loss_log_step)
                loss_log_step = []
                if is_main_process and step > 1 and (not is_debug) and use_wandb:
                    wandb.log({
                        "train_loss": loss_mean,
                        "lr": lr_scheduler.get_lr()[0],
                        "epoch": actual_steps // checkpointing_steps,
                        "sample_time": sample_time,
                        "data_wait_time": data_wait_time
                    }, step=int(actual_steps))
                    zero_rank_print(f"train_loss {loss.item() * gradient_accumulation_steps} epoch {epoch} sample_time {sample_time} data_wait_time {data_wait_time}", LogType.debug)

                zero_rank_print(f"Reset gradients at the beginning of the accumulation cycle", LogType.debug)
                optimizer.zero_grad()
            del loss
            torch.cuda.empty_cache()

            # Update the progress bar
            progress_bar.set_postfix(loss=loss_mean)
            progress_bar.update()
            global_step += 1
            ### <<<< Training <<<< ###

            # Save checkpoint
            print(f"Actual steps {actual_steps} {actual_steps % checkpointing_steps}")
            if is_main_process and actual_steps % checkpointing_steps == 0 and actual_steps > 1:
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.module.state_dict(),
                }
                torch.save(state_dict, f"{run_name}_animatediff.pth")
                # torch.save(state_dict, f"{run_name}_animatediff_epoch_{epoch}.pth")
                wandb.save(f"animatediff_{run_name}_epoch_{epoch}.pth")
                logging.info(f"Saved state to checkpoints {run_name} (global_step: {global_step})")

        # Additional cleanup at the end of an epoch if applicable
        gc.collect()  # Force garbage collection

    dist.destroy_process_group()


