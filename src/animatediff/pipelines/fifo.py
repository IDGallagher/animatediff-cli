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
from tqdm import trange
from tqdm.rich import tqdm
from transformers import CLIPImageProcessor, CLIPTokenizer

from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines import AnimationPipeline, AnimationPipelineOutput
from animatediff.pipelines.context import (get_context_scheduler,
                                           get_total_steps)
from animatediff.utils.model import nop_train
from animatediff.utils.util import get_tensor_interpolation_method, save_video
from ip_adapter import IPAdapter, IPAdapterPlus

logger = logging.getLogger(__name__)

def prepare_fifo_latents(video, scheduler, lookahead_denoising:bool=True):
    latents_list = []
    context_frames = video.shape[2]
    num_inference_steps = scheduler.timesteps.shape[0]

    timesteps = scheduler.timesteps
    timesteps = torch.flip(timesteps, dims=[0])

    if lookahead_denoising:
        for i in range(context_frames // 2):
            noise = torch.randn_like(video[:,:,[0]])
            latents = scheduler.add_noise(video[:,:,[0]], noise, timesteps[0])
            latents_list.append(latents)

    for i in range(num_inference_steps):
        frame_idx = max(0, i-(num_inference_steps - context_frames))
        noise = torch.randn_like(video[:,:,[frame_idx]])
        latents = scheduler.add_noise(video[:,:,[frame_idx]], noise, timesteps[i])
        latents_list.append(latents)

    return torch.cat(latents_list, dim=2)

# def prepare_fifo_latents(video, scheduler, lookahead_denoising:bool=False):
#     latents_list = []
#     context_frames = video.shape[2]
#     num_inference_steps = scheduler.timesteps.shape[0]
#     if lookahead_denoising:
#         for i in range(context_frames // 2):
#             t = scheduler.timesteps[-1]
#             alpha = scheduler.alphas_cumprod[t]
#             beta = 1 - alpha
#             x_0 = video[:,:,[0]]
#             latents = alpha**(0.5) * x_0 + beta**(0.5) * torch.randn_like(x_0)
#             latents_list.append(latents)
#         for i in range(num_inference_steps):
#             t = scheduler.timesteps[num_inference_steps-i-1]
#             alpha = scheduler.alphas_cumprod[t]
#             beta = 1 - alpha
#             frame_idx = max(0, i-(num_inference_steps - context_frames))
#             x_0 = video[:,:,[frame_idx]]

#             latents = alpha**(0.5) * x_0 + beta**(0.5) * torch.randn_like(x_0)
#             latents_list.append(latents)
#     else:
#         for i in range(num_inference_steps):

#             t = scheduler.timesteps[num_inference_steps-i-1]
#             alpha = scheduler.alphas_cumprod[t]
#             beta = 1 - alpha

#             frame_idx = max(0, i-(num_inference_steps - context_frames))
#             x_0 = video[:,:,[frame_idx]]
#             print(f"{i} - t {t} frame_idx {frame_idx} x0 {x_0.shape}")
#             latents = alpha**(0.5) * x_0 + beta**(0.5) * torch.randn_like(x_0)
#             latents_list.append(latents)

#     latents = torch.cat(latents_list, dim=2)

#     return latents

def shift_latents(latents, scheduler):
    # shift latents
    latents[:,:,:-1] = latents[:,:,1:].clone()

    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1]) * scheduler.init_noise_sigma

    return latents

class FifoPipeline(AnimationPipeline):

    def decode_latents(self, latents: torch.Tensor, decode_batch_size: int = 8):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        # video = self.vae.decode(latents).sample

        video = []
        for i in range(0, batch_size * num_frames, decode_batch_size):
            # print(f"Decoding Latent Batch {i}")
            batch_latents = latents[i : i + decode_batch_size]
            batch_video = self.vae.decode(batch_latents.to(self.vae.device, self.vae.dtype)).sample
            video.append(batch_video)
        # for frame_idx in range(latents.shape[0]):
        #     video.append(
        #         self.vae.decode(latents[frame_idx : frame_idx + 1].to(self.vae.device, self.vae.dtype)).sample
        #     )

        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=num_frames)

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
        eta: float = 0.5,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
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
        num_partitions: int = 2,
        **kwargs,
    ):
        num_inference_steps = context_frames * num_partitions # force number of inference steps to be size of queue

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
        sequential_mode = video_length > 24

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # Define call parameters
        batch_size = 1
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

        def get_frame_embeds(context: List[int] = None, video_length : int = 0):
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
        init_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            context_frames,
            height,
            width,
            prompt_embeds.dtype,
            latents_device,  # keep latents on cpu for sequential mode
            generator,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loops
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        print(f"timesteps {timesteps} warmup {num_warmup_steps}")
        print(f"init_latents shape {init_latents.shape}")
        print(f"video_length {video_length}")

        lookahead_denoising = True
        indices = list(range(context_frames * num_partitions))
        print(f"Init indicies {indices}")

        # # 7.1 First 16 frames done in the conventional way
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            progress_bar.set_description("intial sampling")
            context = list(range(context_frames))
            for i, t in enumerate(timesteps):
                print(f"timestep {t}")
                # Expand the latents if doing cfg
                # latent model input torch.Size([2, 4, 16, 64, 64])
                latent_model_input = torch.cat([init_latents] * 2) if do_classifier_free_guidance else init_latents
                # Let the noise scheduler scale the latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(self.unet.device, self.unet.dtype)
                # Get the text and image embeds for this context
                cur_prompt = get_frame_embeds(context, init_latents.shape[2])
                print(f"cur_prompt {cur_prompt.shape}")
                print(f"input_latents {latent_model_input.shape}")
                # Predict the noise for each frame in this context at timestep t
                noise_pred = self.unet(
                    latent_model_input,
                    t, # torch.arange(t, t + 64, step=4, device=self.unet.device)
                    encoder_hidden_states=cur_prompt,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.to(dtype=init_latents.dtype, device=init_latents.device)

                # Combine noise for cfg
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update the latents with the noise prediction
                output_latents = []
                for j, t_j in enumerate(context):
                    output_latents.append(
                        self.scheduler.step(
                            model_output=noise_pred[:, :, [j], :, :],
                            timestep=t,
                            sample=init_latents[:, :, [j], :, :].to(latents_device),
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]
                    )
                init_latents = torch.cat(output_latents, dim=2)
                progress_bar.update()

        # init_latents = torch.load("init_latents.pt").to(latents_device)

        video_out = torch.from_numpy(self.decode_latents(init_latents))
        save_video(video_out, "test.mp4")

        # 7.2 FIFO Denoising
        latents = prepare_fifo_latents(init_latents, self.scheduler, lookahead_denoising=lookahead_denoising)

        video_out = torch.from_numpy(self.decode_latents(latents))
        save_video(video_out, "prepared.mp4")

        self.scheduler.set_timesteps(num_inference_steps, device=latents_device)
        timesteps = self.scheduler.timesteps
        timesteps = torch.flip(timesteps, dims=[0])

        if lookahead_denoising:
            timesteps = torch.cat([torch.full((context_frames//2,), timesteps[0]).to(timesteps.device), timesteps])
            # timesteps = torch.cat([timesteps[0]] * (context_frames // 2) + list(timesteps))
            indices = [0] * (context_frames // 2) + list(indices)

        video = []
        for i in trange(video_length + num_inference_steps - context_frames, desc="fifo sampling"):
            context = list(range(i, i + context_frames))
            context = [min(video_length - 1, max(0, x - context_frames)) for x in context]
            print(f"context {context}")
            for rank in reversed(range(2 * num_partitions if lookahead_denoising else num_partitions)):
                start_idx = rank*(context_frames // 2) if lookahead_denoising else rank*context_frames
                midpoint_idx = start_idx + context_frames // 2
                end_idx = start_idx + context_frames

                t = timesteps[start_idx:end_idx]
                idx = indices[start_idx:end_idx]

                print(f"i {i} rank {rank} start_idx {start_idx} midpoint_idx {midpoint_idx} end_idx {end_idx} - {t} - {idx}")
                input_latents = latents[:,:,start_idx:end_idx].clone()
                print(f"latents {latents.shape}")
                print(f"input_latents {input_latents.shape}")

                # Get the latents corresponding to context window and expand them if doing cfg
                # latent model input torch.Size([2, 4, 16, 64, 64])
                input_latents = torch.cat([input_latents] * 2) if do_classifier_free_guidance else input_latents
                # Let the noise scheduler scale the latents
                input_latents = self.scheduler.scale_model_input(input_latents, t).to(self.unet.device, self.unet.dtype)

                cur_prompt = get_frame_embeds(context, video_length).to(self.unet.device, self.unet.dtype)
                print(f"cur_prompt {cur_prompt.shape}")

                # UNET
                noise_pred = self.unet(
                        input_latents,
                        rearrange(t, 'n -> 1 n').to(self.unet.device),
                        encoder_hidden_states=cur_prompt,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred.to(dtype=latents.dtype, device=latents.device)

                # Combine noise for cfg
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                print(f"noise pred {noise_pred.shape}")
                print(f"noise pred slice {noise_pred[:, :, [0], :, :].shape}")
                # Update the latents with the noise prediction
                output_latents = []
                for j, t_j in enumerate(t):
                    print(f"timestep {t_j}")
                    output_latents.append(
                        self.scheduler.step(
                            model_output=noise_pred[:, :, [j], :, :],
                            timestep=t_j,
                            sample=latents[:, :, [start_idx + j], :, :].to(latents_device),
                            **extra_step_kwargs
                        ).prev_sample
                    )
                print(f"output_latents list {output_latents[0].shape}")
                output_latents = torch.cat(output_latents, dim=2)
                print(f"output latents {output_latents.shape}")

                if lookahead_denoising:
                    latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(context_frames//2):]
                else:
                    latents[:,:,start_idx:end_idx] = output_latents
                del output_latents

            first_frame_idx = context_frames // 2 if lookahead_denoising else 0
            finished_latent = latents[:,:,[first_frame_idx],...]
            if not output_type == "latent":
                decoded_frame = self.decode_latents(finished_latent, decode_batch_size=1)
                video.append(decoded_frame)
            else:
                video.append(finished_latent)
            print(f"Video length {len(video)}")
            # Shift latents along in the queue
            latents = shift_latents(latents, self.scheduler)

        video = np.concatenate(video[-video_length:], axis=2)
        print(f"video {video.shape}")
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
