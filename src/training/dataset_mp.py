import io
import logging
import random
import sys
from functools import partial
from typing import Callable

import decord
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import cpu, gpu
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import CLIPImageProcessor

sys.path.append("./webdataset/")
import wids as wids  # type: ignore

import webdataset as wds
from training.utils import LogType, zero_rank_partial

decord.bridge.set_bridge('torch')
# video_reader = decord.VideoReader("/workspace/animatediff-cli/data/boxer.mp4", ctx=cpu(0))

logger = logging.getLogger(__name__)
zero_rank_print: Callable[[str, LogType], None] = partial(zero_rank_partial, logger)

def make_sample(sample, sample_size=224, target_fps=8, sample_n_frames=16, is_image=False, **kwargs):
    try:
        video_path = sample["mp4"]
        caption = sample["txt"]
        zero_rank_print(f"Loading {caption}", LogType.debug)

        video_reader = decord.VideoReader(io.BytesIO(video_path), ctx=cpu(0))
        # global video_reader

        video_length = len(video_reader)
        zero_rank_print(f"Video length {video_length}", LogType.debug)

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
            # start_frame = 0
            batch_index = [round(start_frame + i * frame_interval) for i in range(sample_n_frames)]
            batch_index = [min(i, video_length - 1) for i in batch_index]

        # Assuming get_batch() returns a PyTorch tensor
        frames = video_reader.get_batch(batch_index)
        # frames = frames.float() / 255.0  # Normalize the pixel values if they're in the 0-255 range

        zero_rank_print(f"Video frames read {batch_index}", LogType.debug)

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

        zero_rank_print(f"Video frames processed", LogType.debug)

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

    return pixel_values, caption

def make_dataloader(shards, batch_size=1, num_workers=1, epoch_size=1000, **kwargs):
    assert(epoch_size % batch_size == 0, f"Make epoch_size {epoch_size} divisible by batch_size {batch_size}")

    dataset = wds.WebDataset(shards, resampled=True, nodesplitter=wds.split_by_node) # , cache_dir="./tmp"
    dataset = dataset.map(partial(make_sample, **kwargs))

    # For IterableDataset objects, the batching needs to happen in the dataset.
    dataset = dataset.batched(batch_size)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    dataloader = dataloader.unbatched().batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    dataloader = dataloader.with_epoch(epoch_size // batch_size)

    return dataloader
