import logging
import re
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPImageProcessor, CLIPTokenizer

from animatediff import get_dir
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.schedulers import get_scheduler
from animatediff.settings import InferenceConfig, ModelConfig
from animatediff.utils.model import (ensure_motion_modules,
                                     get_checkpoint_weights)
from animatediff.utils.util import save_frames, save_images, save_video
from motion_predictor.motion_predictor import MotionPredictor

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
default_base_path = data_dir.joinpath("models/huggingface/stable-diffusion-v1-5")

re_clean_prompt = re.compile(r"[^\w\-, ]")


def create_pipeline(
    base_model: Union[str, PathLike] = default_base_path,
    model_config: ModelConfig = ...,
    infer_config: InferenceConfig = ...,
    use_xformers: bool = True,
) -> AnimationPipeline:
    """Create an AnimationPipeline from a pretrained model.
    Uses the base_model argument to load or download the pretrained reference pipeline model."""

    logger.info("Loading pipeline components...")

    # make sure motion_module is a Path and exists
    logger.debug("Checking motion module...")
    motion_module = data_dir.joinpath(model_config.motion_module)
    if not (motion_module.exists() and motion_module.is_file()):
        # check for safetensors version
        motion_module = motion_module.with_suffix(".safetensors")
        if not (motion_module.exists() and motion_module.is_file()):
            # download from HuggingFace Hub if not found
            ensure_motion_modules()
        if not (motion_module.exists() and motion_module.is_file()):
            # this should never happen, but just in case...
            raise FileNotFoundError(f"Motion module {motion_module} does not exist or is not a file!")

    logger.debug("Loading tokenizer...")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    logger.debug("Loading text encoder...")
    text_encoder: CLIPSkipTextModel = CLIPSkipTextModel.from_pretrained(base_model, subfolder="text_encoder")
    logger.debug("Loading VAE...")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    logger.debug("Loading UNet...")
    unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path=base_model,
        motion_module_path=motion_module,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")

    # set up scheduler
    sched_kwargs = infer_config.noise_scheduler_kwargs
    scheduler = get_scheduler(model_config.scheduler, sched_kwargs)
    logger.info(f'Using scheduler "{model_config.scheduler}" ({scheduler.__class__.__name__})')

    # Load the checkpoint weights into the pipeline
    if model_config.path is not None:
        # Resolve the input model path
        model_path = Path(model_config.path).resolve()
        if model_path.exists():
            # if the absolute model path exists, use it unmodified
            logger.info(f"Loading weights from {model_path}")
        else:
            # otherwise search for the model path relative to the data directory
            model_path = data_dir.joinpath(model_config.path).resolve()
            logger.info(f"Loading weights from {model_path}")

        # Identify whether we have a checkpoint or a HF data dir and load appropriately
        if model_path.is_file():
            logger.debug("Loading from single checkpoint file")
            unet_state_dict, tenc_state_dict, vae_state_dict = get_checkpoint_weights(model_path)
        elif model_path.is_dir():
            logger.debug("Loading from Diffusers model directory")
            temp_pipeline = StableDiffusionPipeline.from_pretrained(model_path)
            unet_state_dict, tenc_state_dict, vae_state_dict = (
                temp_pipeline.unet.state_dict(),
                temp_pipeline.text_encoder.state_dict(),
                temp_pipeline.vae.state_dict(),
            )
            del temp_pipeline
        else:
            raise FileNotFoundError(f"model_path {model_path} is not a file or directory")

        # Load into the unet, TE, and VAE
        logger.info("Merging weights into UNet...")
        _, unet_unex = unet.load_state_dict(unet_state_dict, strict=False)
        if len(unet_unex) > 0:
            raise ValueError(f"UNet has unexpected keys: {unet_unex}")
        tenc_missing, _ = text_encoder.load_state_dict(tenc_state_dict, strict=False)
        if len(tenc_missing) > 0:
            raise ValueError(f"TextEncoder has missing keys: {tenc_missing}")
        vae_missing, _ = vae.load_state_dict(vae_state_dict, strict=False)
        if len(vae_missing) > 0:
            raise ValueError(f"VAE has missing keys: {vae_missing}")
    else:
        logger.info("Using base model weights (no checkpoint/LoRA)")

    # enable xformers if available
    if use_xformers:
        logger.info("Enabling xformers memory-efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    # I'll deal with LoRA later...

    logger.info("Creating AnimationPipeline...")
    pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
    )

    # Load TI embeddings
    load_text_embeddings(pipeline)

    return pipeline


def run_inference(
    pipeline: AnimationPipeline,
    prompt: Optional[str] = None,
    prompt_map: Optional[dict[int, str]] = None,
    n_prompt: str = ...,
    seed: int = -1,
    steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    duration: int = 16,
    idx: int = 0,
    out_dir: PathLike = ...,
    context_frames: int = -1,
    context_stride: int = 3,
    context_overlap: int = 4,
    context_schedule: str = "uniform",
    context_loop: bool = False,
    clip_skip: int = 1,
    return_dict: bool = False,
    video_tensor: Optional[torch.FloatTensor] = None,
    input_images: Optional[dict[int, str]] = None,
    interpolate_images: bool = False,
):
    if prompt is None and prompt_map is None:
        raise ValueError("prompt and prompt_map cannot both be None, one must be provided")

    out_dir = Path(out_dir)  # ensure out_dir is a Path

    if seed != -1:
        torch.manual_seed(seed)
    else:
        seed = torch.seed()

    pos_image_embeds, neg_image_embeds, image_embed_frames = None, None, []
    if video_tensor is not None:
        pipeline.load_ip_adapter(scale=0.5)
        # Normalize tensor if necessary (this depends on how it was preprocessed)
        # Check if normalization is needed (if values are not in [0, 1])
        if video_tensor.min() < 0 or video_tensor.max() > 1:
            video_tensor = (video_tensor + 1) / 2  # Adjust from [-1, 1] to [0, 1] for PIL

        logger.debug(f"Video tensor shape {video_tensor.shape}")

        # Convert frames to PIL images
        pil_images = [to_pil_image(frame) for frame in video_tensor]

        # Get image embeddings using the provided IP adapter method
        pos_image_embeds, neg_image_embeds = pipeline.ip_adapter.get_image_embeds(pil_images)
        image_embed_frames = range(pos_image_embeds.shape[0])
        logger.debug(f"Created image embeds {pos_image_embeds.shape} {neg_image_embeds.shape}")
    elif input_images is not None:
        pipeline.load_ip_adapter(scale=0.8)
        pil_images = []

        # Load and convert each image file to a PIL Image
        for idx, file_path in input_images.items():
            img = Image.open(file_path)
            # Assume images are already normalized to [0, 1], add any additional necessary preprocessing if required
            pil_images.append(img)
        # Save initial
        save_images(pil_images, out_dir.joinpath(f"initial"))
        # Get image embeddings using the provided IP adapter method
        pos_image_embeds, neg_image_embeds = pipeline.ip_adapter.get_image_embeds(pil_images)

        if interpolate_images:
            logger.debug(f"Interpolating to create {duration} image embeds {pipeline.device}")
            with torch.inference_mode(True):
                motion_predictor = MotionPredictor().to(pipeline.device, dtype=torch.float16)

                checkpoint = torch.load("outputs/training_mp-2024-07-21T03-46-53/checkpoints/motion_predictor_epoch_38.pth")
                # Load the state dictionary into the model
                motion_predictor.load_state_dict(checkpoint)
                logger.debug(f"pos_image_embeds {pos_image_embeds.shape}")
                pos_image_embeds = pos_image_embeds.unsqueeze(0)  # Add batch dimension, shape: (1, sequence_length, feature_dim)
                pos_image_embeds = motion_predictor(pos_image_embeds[:, 0], pos_image_embeds[:, -1], total_frames=duration).squeeze(0)
                # pos_image_embeds = motion_predictor.interpolate_tokens(pos_image_embeds[:, 0, :], pos_image_embeds[:, -1, :]).squeeze(0)
                neg_image_embeds = neg_image_embeds.unsqueeze(0)
                # neg_image_embeds = motion_predictor(neg_image_embeds[:, 0], neg_image_embeds[:, -1]).squeeze(0)
                neg_image_embeds = motion_predictor.interpolate_tokens(neg_image_embeds[:, 0], neg_image_embeds[:, -1], total_frames=duration).squeeze(0)
                image_embed_frames = range(pos_image_embeds.shape[0])
                logger.debug(f"pos embeds shape {pos_image_embeds.shape}")
                logger.debug(f"neg embeds shape {neg_image_embeds.shape}")
        else:
            image_embed_frames = [int(idx) for idx in input_images.keys()]  # Save indices of the frames
            logger.debug(f"Processed image embeds for {len(pil_images)} images")

    with torch.inference_mode(True):
        pipeline_output = pipeline(
            prompt=prompt,
            prompt_map=prompt_map,
            negative_prompt=n_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            video_length=duration,
            return_dict=return_dict,
            context_frames=context_frames,
            context_stride=context_stride + 1,
            context_overlap=context_overlap,
            context_schedule=context_schedule,
            context_loop=context_loop,
            clip_skip=clip_skip,
            pos_image_embeds=pos_image_embeds,
            neg_image_embeds=neg_image_embeds,
            image_embed_frames=image_embed_frames
        )
    logger.info("Generation complete, saving...")

    # Trim and clean up the prompt for filename use
    prompt_str = prompt or next(iter(prompt_map.values()))
    prompt_tags = [re_clean_prompt.sub("", tag).strip().replace(" ", "-") for tag in prompt_str.split(",")]
    prompt_str = "_".join((prompt_tags[:6]))

    # generate the output filename and save the video
    out_str = f"{idx:02d}_{seed}_{prompt_str}"[:250]
    out_file = out_dir.joinpath(f"{out_str}.mp4")
    if return_dict is True:
        save_video(pipeline_output["videos"], out_file)
    else:
        save_video(pipeline_output, out_file)

    logger.info(f"Saved sample to {out_file}")
    return pipeline_output
