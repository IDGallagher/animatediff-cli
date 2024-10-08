# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (DDIMScheduler, DPMSolverMultistepScheduler,
                                  EulerAncestralDiscreteScheduler,
                                  EulerDiscreteScheduler, LMSDiscreteScheduler,
                                  PNDMScheduler)
from diffusers.utils import (BaseOutput, deprecate, is_accelerate_available,
                             is_accelerate_version)
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from packaging import version
from tqdm.rich import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.context import (get_context_scheduler,
                                           get_total_steps)
from animatediff.utils.model import nop_train
from animatediff.utils.util import get_tensor_interpolation_method
from ip_adapter import IPAdapter, IPAdapterPlus

logger = logging.getLogger(__name__)


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    _optional_components = ["feature_extractor"]

    vae: AutoencoderKL
    text_encoder: CLIPSkipTextModel
    tokenizer: CLIPTokenizer
    unet: UNet3DConditionModel
    feature_extractor: CLIPImageProcessor
    scheduler: Union[
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    ]
    ip_adapter: IPAdapter = None

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPSkipTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: int = 1,
    ):
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
                clip_skip=clip_skip,
            )
            prompt_embeds = prompt_embeds[0]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_tokens: list[str]
            if negative_prompt is None:
                negative_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                negative_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                negative_tokens = self.maybe_convert_prompt(negative_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            neg_input_ids = self.tokenizer(
                negative_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = neg_input_ids.input_ids

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = neg_input_ids.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids.to(device),
                attention_mask=attention_mask,
                clip_skip=clip_skip,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def load_ip_adapter(self, is_plus:bool=True, scale:float=1.0):
        if self.ip_adapter is None:
            img_enc_path = "data/models/CLIP-ViT-H-14-laion2B-s32B-b79K"

            if is_plus:
                self.ip_adapter = IPAdapterPlus(self, img_enc_path, "data/models/IP-Adapter/models/ip-adapter-plus_sd15.bin", self.device, 16)
            else:
                self.ip_adapter = IPAdapter(self, img_enc_path, "data/models/IP-Adapter/models/ip-adapter_sd15.bin", self.device, 4)
            self.ip_adapter.set_scale(scale)

    def unload_ip_adapater(self):
        if self.ip_adapter:
            self.ip_adapter.unload()
            self.ip_adapter = None
            torch.cuda.empty_cache()

    def decode_latents(self, latents: torch.Tensor):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(
                self.vae.decode(latents[frame_idx : frame_idx + 1].to(self.vae.device, self.vae.dtype)).sample
            )
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.unet.device, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents.to(device, dtype)

    def __call__(
        self,
        prompt: Optional[str] = None,
        prompt_map: Optional[dict[int, str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, list[str]]] = None,
        video_length: int = ...,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        context_frames: int = 16,
        context_stride: int = 1,
        context_overlap: int = 4,
        context_schedule: str = "uniform",
        context_loop: bool = False,
        clip_skip: int = 1,
        pos_image_embeds: Optional[torch.FloatTensor] = None,
        neg_image_embeds: Optional[torch.FloatTensor] = None,
        image_embed_frames: list[int] = [],
        is_single_prompt_mode: bool = False,
        **kwargs,
    ):
        if prompt is None and prompt_map is None:
            raise ValueError("Must provide a prompt or a prompt map.")

        if prompt_map is None:
            prompt_map = {0: prompt}

        # prepare map for prompt travel by frame index
        last_frame = video_length - 1
        prompt_map = self._prepare_map(prompt_map, last_frame)
        if len(prompt_map) > 1:
            logger.info("Keyframes for this animation:")
            for idx, prompt in prompt_map.items():
                logger.info(f" - F{idx:03d}: {prompt[:70]}...")

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 24 frames is max reliable number for one-shot mode, so we use sequential mode for longer videos
        # sequential_mode = video_length > 24
        sequential_mode = False

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # Define call parameters
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        latents_device = kwargs.pop("latents_device", None)
        if latents_device is None:
            latents_device = torch.device("cpu") if sequential_mode else device
        else:
            latents_device = torch.device(latents_device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        ### text
        prompt_embeds_map = {}
        prompt_map = dict(sorted(prompt_map.items()))

        prompt_list = [prompt_map[key_frame] for key_frame in prompt_map.keys()]
        prompt_embeds = self._encode_prompt(
            prompt_list,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            clip_skip=clip_skip,
        )

        if do_classifier_free_guidance:
            negative, positive = prompt_embeds.chunk(2, 0)
            negative = negative.chunk(negative.shape[0], 0)
            positive = positive.chunk(positive.shape[0], 0)
        else:
            positive = prompt_embeds
            positive = positive.chunk(positive.shape[0], 0)

        if self.ip_adapter:
            self.ip_adapter.set_text_length(positive[0].shape[1])

        for i, key_frame in enumerate(prompt_map):
            if do_classifier_free_guidance:
                prompt_embeds_map[key_frame] = torch.cat([negative[i] , positive[i]])
            else:
                prompt_embeds_map[key_frame] = positive[i]

        key_first =list(prompt_map.keys())[0]
        key_last =list(prompt_map.keys())[-1]

        def get_current_prompt_embeds_from_text(
                center_frame = None,
                video_length : int = 0
                ):

            key_prev = key_last
            key_next = key_first

            for p in prompt_map.keys():
                if p > center_frame:
                    key_next = p
                    break
                key_prev = p

            dist_prev = center_frame - key_prev
            if dist_prev < 0:
                dist_prev += video_length
            dist_next = key_next - center_frame
            if dist_next < 0:
                dist_next += video_length

            if key_prev == key_next or dist_prev + dist_next == 0:
                return prompt_embeds_map[key_prev]

            rate = dist_prev / (dist_prev + dist_next)

            return get_tensor_interpolation_method()( prompt_embeds_map[key_prev], prompt_embeds_map[key_next], rate )

        ### image
        if self.ip_adapter:
            im_prompt_embeds_map = {}
            ip_im_map = {i: torch.tensor([]) for i in image_embed_frames}

            positive = pos_image_embeds
            negative = neg_image_embeds

            bs_embed, seq_len, _ = positive.shape
            positive = positive.repeat(1, 1, 1)
            positive = positive.view(bs_embed * 1, seq_len, -1)

            bs_embed, seq_len, _ = negative.shape
            negative = negative.repeat(1, 1, 1)
            negative = negative.view(bs_embed * 1, seq_len, -1)

            if do_classifier_free_guidance:
                negative = negative.chunk(negative.shape[0], 0)
                positive = positive.chunk(positive.shape[0], 0)
            else:
                positive = positive.chunk(positive.shape[0], 0)

            for i, key_frame in enumerate(ip_im_map):
                if do_classifier_free_guidance:
                    im_prompt_embeds_map[key_frame] = torch.cat([negative[i] , positive[i]])
                else:
                    im_prompt_embeds_map[key_frame] = positive[i]

            im_key_first =list(ip_im_map.keys())[0]
            im_key_last =list(ip_im_map.keys())[-1]

        def get_current_prompt_embeds_from_image(
                center_frame = None,
                video_length : int = 0
                ):

            key_prev = im_key_last
            key_next = im_key_first

            for p in ip_im_map.keys():
                if p > center_frame:
                    key_next = p
                    break
                key_prev = p

            dist_prev = center_frame - key_prev
            if dist_prev < 0:
                dist_prev += video_length
            dist_next = key_next - center_frame
            if dist_next < 0:
                dist_next += video_length

            if key_prev == key_next or dist_prev + dist_next == 0:
                return im_prompt_embeds_map[key_prev]

            rate = dist_prev / (dist_prev + dist_next)

            return get_tensor_interpolation_method()( im_prompt_embeds_map[key_prev], im_prompt_embeds_map[key_next], rate)

        def get_frame_embeds(
                context: List[int] = None,
                video_length : int = 0
                ):

            neg = []
            pos = []
            for c in context:
                t = get_current_prompt_embeds_from_text(c, video_length)
                if do_classifier_free_guidance:
                    negative, positive = t.chunk(2, 0)
                    neg.append(negative)
                    pos.append(positive)
                else:
                    pos.append(t)

            if do_classifier_free_guidance:
                neg = torch.cat(neg)
                pos = torch.cat(pos)
                text_emb = torch.cat([neg , pos])
            else:
                pos = torch.cat(pos)
                text_emb = pos

            if self.ip_adapter == None:
                return text_emb

            neg = []
            pos = []
            for c in context:
                im = get_current_prompt_embeds_from_image(c, video_length)
                if do_classifier_free_guidance:
                    negative, positive = im.chunk(2, 0)
                    neg.append(negative)
                    pos.append(positive)
                else:
                    pos.append(im)

            if do_classifier_free_guidance:
                neg = torch.cat(neg)
                pos = torch.cat(pos)
                image_emb = torch.cat([neg , pos])
            else:
                pos = torch.cat(pos)
                image_emb = pos

            return torch.cat([text_emb,image_emb], dim=1)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=latents_device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            prompt_embeds.dtype,
            latents_device,  # keep latents on cpu for sequential mode
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.5 - Infinite context loop shenanigans
        context_scheduler = get_context_scheduler(context_schedule)
        total_steps = get_total_steps(
            context_scheduler,
            timesteps,
            num_inference_steps,
            latents.shape[2],
            context_frames,
            context_stride,
            context_overlap,
            context_loop
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=total_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # latents shape torch.Size([1, 4, 24, 64, 64])
                # noise pred shape torch.Size([2, 4, 24, 64, 64])
                # Holder for model's noise prediciton
                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # Count of how many predictions we have for each frame
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
                )

                for context in context_scheduler(
                    i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
                ):
                    # Get the latents corresponding to context window and expand them if doing cfg
                    # latent model input torch.Size([2, 4, 16, 64, 64])
                    latent_model_input = (latents[:, :, context].to(device).repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )

                    # Let the noise scheduler scale the latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(self.unet.device, self.unet.dtype)

                    # Get the text and image embeds for this context
                    cur_prompt = get_frame_embeds(context, latents.shape[2])
                    # print(f"cur_prompt {cur_prompt.shape}")
                    # print(f"input_latents {latent_model_input.shape}")
                    # Predict the noise for each frame in this context at timestep t
                    pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=cur_prompt,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                    pred = pred.to(dtype=latents.dtype, device=latents.device)

                    # Add the predicted noise to any pre-existing predicitons
                    noise_pred[:, :, context] = noise_pred[:, :, context] + pred
                    # Count the predictions
                    counter[:, :, context] = counter[:, :, context] + 1
                    progress_bar.update()

                # Average out any noise that we got more than one of
                noise_pred = noise_pred / counter

                # Combine noise for cfg
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update the latents with the noise prediction
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents.to(latents_device),
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # TEMP HACK
        torch.save(latents, f"init_latents.pt")

        # Return latents if requested (this will never be a dict)
        if not output_type == "latent":
            video = self.decode_latents(latents)
        else:
            video = latents

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def freeze(self):
        logger.debug("Freezing pipeline...")
        _ = self.unet.eval()
        self.unet = self.unet.requires_grad_(False)
        self.unet.train = nop_train

        _ = self.text_encoder.eval()
        self.text_encoder = self.text_encoder.requires_grad_(False)
        self.text_encoder.train = nop_train

        _ = self.vae.eval()
        self.vae = self.vae.requires_grad_(False)
        self.vae.train = nop_train

    def _prepare_map(self, prompt_map: dict[int, str], last_frame: int) -> dict[int, str]:
        # if we only have one prompt, just set its key to 0 and return
        if len(prompt_map) == 1:
            return {0: next(iter(prompt_map.values()))}

        # helper to get current indexes
        def frame_ids():
            return sorted([int(x) for x in prompt_map.keys()])

        # remap -1 to the last frame
        if -1 in frame_ids():
            prompt_map[last_frame] = prompt_map.pop(-1)

        # if no prompt for first frame, copy the first prompt
        if 0 not in frame_ids():
            prompt_map[0] = prompt_map[frame_ids()[0]]

        # if no prompt for last frame, copy the last prompt
        if last_frame not in frame_ids():
            prompt_map[last_frame] = prompt_map[frame_ids()[-1]]

        # make sure our max is not greater than the last frame
        max_frame = max(frame_ids())
        if max_frame > last_frame:
            raise ValueError(f"Prompt map has a frame {max_frame} greater than the last frame {last_frame}")

        # sort the prompt map by frame index and return
        prompt_map = dict(sorted(prompt_map.items()))
        return prompt_map
