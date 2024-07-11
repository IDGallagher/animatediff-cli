import random
import sys
from functools import partial

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

sys.path.append("./webdataset/")
import wids as wids  # type: ignore

import webdataset as wds


def make_sample(sample, sample_size=256, target_fps=8, sample_n_frames=16, is_image=False, **kwargs):
    try:
        video_path = sample[".mp4"]
        caption = sample[".txt"]

        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        if video_length == 0:
            raise ValueError("Empty video file")

        original_fps = video_reader.get_avg_fps()

        if is_image:
            batch_index = [random.randint(0, video_length - 1)]
        else:
            frame_interval = original_fps / target_fps
            total_duration = (sample_n_frames - 1) / target_fps
            max_start_frame = max(0, video_length - int(total_duration * original_fps) - 1)
            start_frame = random.randint(0, max_start_frame)
            batch_index = [round(start_frame + i * frame_interval) for i in range(sample_n_frames)]
            batch_index = [min(i, video_length - 1) for i in batch_index]  # Ensure we don't exceed video length

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if is_image:
            pixel_values = pixel_values[0]

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        pixel_transforms = transforms.Compose([
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    except KeyError as e:
        print(f"Missing key in sample: {e}")
        return None
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error processing sample: {e}")
        return None

    return dict(pixel_values=pixel_transforms(pixel_values), text=caption)

def make_dataset(shards, cache_dir="./tmp", **kwargs):
    trainset = wids.ShardListDataset(shards, cache_dir=cache_dir, keep=True)
    trainset = trainset.add_transform(partial(make_sample, **kwargs))
    return trainset

def make_dataloader(dataset, batch_size=1):
    sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=1
    )
    return dataloader
