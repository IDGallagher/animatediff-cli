import random
import sys
from functools import partial

import decord
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import CLIPImageProcessor

sys.path.append("./webdataset/")
import wids as wids  # type: ignore

import webdataset as wds

decord.bridge.set_bridge('torch')

def make_sample(sample, sample_size=224, target_fps=8, sample_n_frames=16, is_image=False, **kwargs):
    try:
        video_path = sample[".mp4"]
        caption = sample[".txt"]

        video_reader = decord.VideoReader(video_path)
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
            batch_index = [min(i, video_length - 1) for i in batch_index]

        # Assuming get_batch() returns a PyTorch tensor
        frames = video_reader.get_batch(batch_index)
        frames = frames.float() / 255.0  # Normalize the pixel values if they're in the 0-255 range

        # Initialize the CLIPImageProcessor with the appropriate configuration
        clip_processor = CLIPImageProcessor(
            do_resize=True,
            size={"shortest_edge": sample_size},
            do_center_crop=True,
            crop_size={"height": sample_size, "width": sample_size},
            do_normalize=True
        )

        # Process the batch of frames directly
        processed_frames = clip_processor.preprocess(
            images=frames,
            return_tensors='pt'
        )['pixel_values']

        if is_image:
            pixel_values = processed_frames[0]  # Just use the first frame
        else:
            pixel_values = processed_frames

        del video_reader

    except KeyError as e:
        print(f"Missing key in sample: {e}")
        return None
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"Error processing sample: {e}")
        return None

    return dict(pixel_values=pixel_values, text=caption)

def make_dataset(shards, cache_dir="./tmp", **kwargs):
    trainset = wids.ShardListDataset(shards, cache_dir=cache_dir, keep=True)
    trainset = trainset.add_transform(partial(make_sample, **kwargs))
    return trainset

def make_dataloader(dataset, batch_size=1, num_workers=1):
    sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    return dataloader
