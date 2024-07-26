import datetime
import inspect
import logging
import math
import os
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler as get_lr_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import wandb
from animatediff.utils.util import save_frames, save_video
from ip_adapter import IPAdapter, IPAdapterPlus
from motion_predictor.motion_predictor import MotionPredictor
from training.dataset_mp import make_dataloader

from .utils import LogType, zero_rank_partial

logger = logging.getLogger(__name__)
zero_rank_print: Callable[[str, LogType], None] = partial(zero_rank_partial, logger)


def load_ip_adapter(sd_model_path:str, is_plus:bool=True, scale:float=1.0, device='cpu'):
    img_enc_path = "data/models/CLIP-ViT-H-14-laion2B-s32B-b79K"

    # We're just using this pipeline to get unet parameters. It can be safely deleted once we're intialized
    temp_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_path)

    if is_plus:
        ip_adapter = IPAdapterPlus(temp_pipeline, img_enc_path, "data/models/IP-Adapter/models/ip-adapter-plus_sd15.bin", device, 16)
    else:
        ip_adapter = IPAdapter(temp_pipeline, img_enc_path, "data/models/IP-Adapter/models/ip-adapter_sd15.bin", device, 4)
    ip_adapter.set_scale(scale)

    # Delete pipeline and return
    ip_adapter.pipe = None
    del temp_pipeline
    torch.cuda.empty_cache()
    return ip_adapter

def train_mp(
        name: str,
        use_wandb: bool,

        output_dir: str,
        sd_model_path: str,

        train_data: Dict,
        validation_data: Dict,
        device_id: int,

        epoch_size:int = 1000,
        num_epochs:int = 1,
        validation_steps: int = 100,
        validation_steps_tuple: Tuple = (-1,),

        learning_rate: float = 3e-5,
        scale_lr: bool = False,
        lr_warmup_steps: int = 0,
        lr_scheduler: str = "constant",

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

    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="motionpredictor", name=folder_name, config=config)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load models and move to GPU
    ip_adapter = load_ip_adapter(sd_model_path, is_plus=True, device=device_id)
    model = MotionPredictor().to(device=device_id)

    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.kaiming_uniform_(m.weight)
    #         if m.bias is not None:
    #             torch.nn.init.constant_(m.bias, 0)
    # model.apply(init_weights)

    # Freeze IPA
    # ip_adapter.requires_grad_(False)

    # Set unet trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    zero_rank_print(f"trainable params number: {len(trainable_params)}")
    zero_rank_print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # enable xformers if available
    if enable_xformers_memory_efficient_attention:
        logger.info("Enabling xformers memory-efficient attention")
        model.enable_xformers_memory_efficient_attention()

    # Enable gradient checkpointing
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.enable_gradient_checkpointing()

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    criterion = nn.MSELoss()

    # Prepare the data loader
    train_dataloader = make_dataloader(**train_data, batch_size=train_batch_size, num_workers=num_workers)

    # # Get the training iteration
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

    # DDP Wrapper
    model = DDP(model, device_ids=[device_id], output_device=device_id)

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
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    # DUMMY
    # batches = []
    # for step, batch in enumerate(train_dataloader):
        # zero_rank_print(f"step {step}")
        # batches.append(batch)
        # break
    # batches *= epoch_size
    # End DUMMY

    # Training loop
    model.train()
    for epoch in range(first_epoch, num_epochs):

        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=epoch_size, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for step, batch in progress_bar:

        # DUMMY
        # for step, batch in enumerate(batches):
        # End DUMMY

            pixel_values = batch[0].to(device_id)
            logger.debug(f"Pixel values {pixel_values.shape}")

            # Data batch sanity check
            if epoch == first_epoch and step < 2:
                sanity_pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                for idx, pixel_value in enumerate(sanity_pixel_values):
                    pixel_value = pixel_value[None, ...]
                    save_frames(pixel_value, f"{output_dir}/sanity_check/{epoch}-{step}-{idx}/")
                    save_video(pixel_value, f"{output_dir}/sanity_check/{epoch}-{step}-{idx}.mp4")

            # Get image embeddings using the provided IP adapter method
            ground_truth = ip_adapter.get_image_embeds_preprocessed(pixel_values)
            logger.debug(f"Ground truth shape {ground_truth.shape}")

            # Forward pass
            zero_rank_print(f"Forward Pass with Mixed Precision: {mixed_precision_training} Total frames: {ground_truth.shape[1]}", LogType.debug)
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                outputs = model(ground_truth[:, 0], ground_truth[:, -1], total_frames=ground_truth.shape[1])
                loss = criterion(outputs, ground_truth) / gradient_accumulation_steps

            # Backpropagate loss, accumulate gradients
            zero_rank_print("Backpropagate", LogType.debug)
            if mixed_precision_training:
                scaler.scale(loss).backward()  # Scale the loss; backward pass accumulates gradients
            else:
                loss.backward()  # Gradient accumulation without scaling

            if (step + 1) % gradient_accumulation_steps == 0:
                zero_rank_print("=== Accumulate gradients", LogType.debug)
                if mixed_precision_training:
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)  # Perform optimizer step
                    scaler.update()  # Update the scale for next iteration
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                lr_scheduler.step()  # Update learning rate

                # Log to WandB
                if is_main_process and (not is_debug) and use_wandb:
                    wandb.log({"train_loss": loss.item() * gradient_accumulation_steps, "epoch": epoch})
                    zero_rank_print(f"train_loss {loss.item() * gradient_accumulation_steps} epoch {epoch}", LogType.debug)
                epoch_loss += loss.item() * gradient_accumulation_steps

            # Update the progress bar
            progress_bar.set_postfix(loss=epoch_loss / (step + 1))
            global_step += 1

        # Save checkpoint
        # if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataset) - 1):
        if is_main_process:
            # Assuming the model is wrapped in DataParallel or DistributedDataParallel
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                torch.save(model.module.state_dict(), f"{output_dir}/checkpoints/motion_predictor_epoch_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"{output_dir}/checkpoints/motion_predictor_epoch_{epoch}.pth")
            wandb.save(f"checkpoint_epoch_{epoch}.pth")

    # Cleanup
    dist.barrier()
    wandb.finish()
