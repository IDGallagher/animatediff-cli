import random
from os import PathLike
from pathlib import Path
from typing import List

import ffmpeg
import PIL
import torch
from decord import VideoReader, cpu, gpu
from einops import rearrange
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.rich import tqdm

tensor_interpolation = None

def get_tensor_interpolation_method():
    return tensor_interpolation

def set_tensor_interpolation_method(is_slerp):
    global tensor_interpolation
    tensor_interpolation = slerp if is_slerp else linear

def linear(v1, v2, t):
    return (1.0 - t) * v1 + t * v2

def slerp(
    v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995
) -> torch.Tensor:
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    dot = (u0 * u1).sum()
    if dot.abs() > DOT_THRESHOLD:
        #logger.info(f'warning: v0 and v1 close to parallel, using linear interpolation instead.')
        return (1.0 - t) * v0 + t * v1
    omega = dot.acos()
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()


def load_video_frames(video_path, sample_n_frames, target_fps, sample_size, is_image=False, device='cpu'):

    video_reader = VideoReader(video_path, ctx=cpu() if device == 'cpu' else gpu(0))
    video_length = len(video_reader)

    if video_length == 0:
        raise ValueError("Empty video file")

    original_fps = video_reader.get_avg_fps()

    if is_image:
        # If treating the video as a single image, pick the first frame
        batch_index = 0
    else:
        # Calculate frames to sample to simulate the target_fps
        frame_interval = original_fps / target_fps
        start_frame = 0
        batch_index = [round(start_frame + i * frame_interval) for i in range(sample_n_frames)]
        batch_index = [min(i, video_length - 1) for i in batch_index]  # Ensure we don't exceed video length

    # Ensure sample_size is a tuple
    sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

    # Define transformations
    pixel_transforms = transforms.Compose([
        transforms.Resize(sample_size[0]),  # Resize the frame
        transforms.CenterCrop(sample_size),  # Crop the frame to the desired size
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # Normalize pixel values
    ])

    # Load and transform frames
    frames = video_reader.get_batch(batch_index).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # Convert to tensor (B, C, H, W)
    frames = frames.float() / 255.0  # Convert to float and normalize
    frames = pixel_transforms(frames)  # Apply transformations

    return frames

def load_video(video_path, length, device='cpu'):
    vr = VideoReader(video_path, ctx=cpu() if device == 'cpu' else gpu(0))
    frames = vr.get_batch(range(length)).asnumpy()  # Get the first 'length' frames
    return torch.tensor(frames).permute(0, 3, 1, 2) / 255.0  # Convert to tensor and normalize


def save_frames(video: Tensor, frames_dir: PathLike):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = rearrange(video, "b c t h w -> t b c h w")
    for idx, frame in enumerate(tqdm(frames, desc=f"Saving frames to {frames_dir.stem}")):
        save_image(frame, frames_dir.joinpath(f"{idx:03d}.png"))

def save_images(images: List[Image.Image], frames_dir: str) -> None:
    # Convert frames_dir to a Path object to utilize pathlib's methods
    frames_dir = Path(frames_dir)

    # Create the directory if it doesn't exist
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Loop through images and save each
    for idx, frame in enumerate(tqdm(images, desc=f"Saving frames to {frames_dir.stem}")):
        # Use the save() method from PIL.Image to save the image
        frame.save(frames_dir.joinpath(f"{idx:03d}.png"))

def save_gif(video: Tensor, save_path: PathLike, fps: int = 8):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim == 5:
        # batch, channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(0, 2, 1, 3, 4).squeeze(0)
    elif video.ndim == 4:
        # channels, frame, width, height -> frame, channels, width, height
        frames = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"video must be 4 or 5 dimensional, got {video.ndim}")

    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    frames = frames.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        fp=save_path, format="GIF", append_images=images[1:], save_all=True, duration=(1 / fps * 1000), loop=0
    )

def save_video(video: torch.Tensor, save_path: str, fps: int = 8):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim == 5:
        # batch, channels, frame, width, height -> frame, channels, width, height
        video = video.squeeze(0)  # Assume a single batch
    if video.ndim == 4:
        # channels, frame, width, height -> frame, width, height, channels
        video = video.permute(1, 2, 3, 0)
    else:
        raise ValueError(f"video must be 4 or 5 dimensional, got {video.ndim}")

    # Convert tensor to numpy array and scale from 0 to 255
    video_np = video.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()

    # Define process inputs for ffmpeg
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(video.shape[2], video.shape[1]), r=fps)
        .output(str(save_path), pix_fmt='yuv420p', vcodec='libx264', r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # Write video frames to stdin
    for frame in video_np:
        process.stdin.write(frame.tobytes())

    # Close the process
    process.stdin.close()
    process.wait()


def relative_path(path: PathLike, base: PathLike = Path.cwd()) -> str:
    path = Path(path).resolve()
    base = Path(base).resolve()
    try:
        relpath = str(path.relative_to(base))
    except ValueError:
        relpath = str(path)
    return relpath
