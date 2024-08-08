import gc
import io
import logging
import random
import sys
from functools import partial
from typing import Callable

import decord
import numpy as np
import torch
import torchaudio
import torchvision.transforms as transforms
from decord import cpu, gpu
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchaudio.io import StreamReader
from torchaudio.utils import ffmpeg_utils
from transformers import CLIPImageProcessor

import webdataset as wds
from training.utils import LogType, zero_rank_partial
from webdataset.handlers import warn_and_continue

decord.bridge.set_bridge('torch')

logger = logging.getLogger(__name__)
logger.disabled = False
zero_rank_print: Callable[[str, LogType], None] = partial(zero_rank_partial, logger)

def make_sample(sample, sample_size=256, target_fps=8, sample_n_frames=16, is_image=False, **kwargs):
    stream_reader = None
    frames = None
    try:
        video_path = sample["mp4"]
        caption = io.BytesIO(sample["txt"]).getvalue().decode('UTF-8')
        zero_rank_print(f"Loading {caption}", LogType.debug)

        if is_image:
            sample_n_frames = 1

        stream_reader = torchaudio.io.StreamReader(io.BytesIO(video_path))
        stream_reader.add_video_stream(
            frames_per_chunk=sample_n_frames,
            filter_desc=f"fps={target_fps},scale={sample_size}:{sample_size}:force_original_aspect_ratio=increase,crop={sample_size}:{sample_size},format=pix_fmts=rgb24"
        )
        # pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()   ????
        metadata = stream_reader.get_src_stream_info(0)
        original_fps = metadata.frame_rate
        video_length = metadata.num_frames
        duration = video_length / original_fps  # Duration in seconds
        zero_rank_print(f"Video length {video_length} at {original_fps} fps", LogType.debug)

        if video_length == 0:
            raise ValueError("Empty video file")

        target_video_length = int(duration * target_fps)

        max_start_frame = max(0, target_video_length - sample_n_frames)
        start_frame = random.randint(0, max_start_frame)

        # Convert start_frame to seconds
        start_time = start_frame / target_fps

        stream_reader.seek(start_time)
        stream_reader.fill_buffer()
        (frames,) = stream_reader.pop_chunks()

        # frames = frames[0]
        zero_rank_print(f"frames {frames.shape} {frames.dtype} {frames.device}", LogType.debug)

        zero_rank_print(f"Video frames read and processed, start frame {start_frame} start time {start_time:.2f}s", LogType.debug)

        if is_image:
            pixel_values = frames[0]
        else:
            pixel_values = frames

        # Check video size
        if not is_image and pixel_values.shape[0] != sample_n_frames:
            zero_rank_print(f"Video length mismatch: expected {sample_n_frames}, got {pixel_values.shape[0]}", LogType.error)
            return None

        # Explicitly delete large objects
        del video_path

    except KeyError as e:
        zero_rank_print(f"Missing key in sample: {e}", LogType.error)
        return None
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        zero_rank_print(f"Error processing sample: {e}", LogType.error)
        return None
    finally:
        # Ensure we close and delete the StreamReader
        if stream_reader is not None:
            stream_reader.remove_stream(0)
            del stream_reader

        # Delete the frames list if it exists
        if frames is not None:
            del frames

        # Force garbage collection
        gc.collect()

    return pixel_values, caption, start_time

def make_dataloader(shards, batch_size, num_workers, epoch_size, **kwargs):
    assert(epoch_size % batch_size == 0, f"Make epoch_size {epoch_size} divisible by batch_size {batch_size}")

    dataset = wds.WebDataset(shards, handler=warn_and_continue, resampled=True, nodesplitter=wds.split_by_node) # , cache_dir="./tmp"
    dataset = dataset.shuffle(1000).map(partial(make_sample, **kwargs), handler=warn_and_continue)

    # For IterableDataset objects, the batching needs to happen in the dataset.
    dataset = dataset.batched(batch_size)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    dataloader = dataloader.unbatched().shuffle(1000).batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    dataloader = dataloader.with_epoch(epoch_size // batch_size)

    return dataloader
