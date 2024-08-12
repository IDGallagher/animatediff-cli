import inspect
import logging
from enum import Enum

import torch
import torch.distributed as dist


class LogType(str, Enum):
    info = "info"
    debug = "debug"
    error = "error"

def zero_rank_partial(logger, s, logtype:LogType = LogType.info):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
        if logtype == LogType.info:
            logger.info(s)
        elif logtype == LogType.debug:
            logger.debug(s)
        elif logtype == LogType.error:
            logger.error(s)

def apply_lora(unet, state_dict, alpha=1.0):
    # directly update weight in diffusers model
    for key in state_dict:
        # only process lora down key
        if "up." in key: continue

        up_key    = key.replace(".down.", ".up.")
        model_key = key.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
        model_key = model_key.replace("to_out.", "to_out.0.")
        layer_infos = model_key.split(".")[:-1]

        curr_layer = unet
        while len(layer_infos) > 0:
            temp_name = layer_infos.pop(0)
            curr_layer = curr_layer.__getattr__(temp_name)

        weight_down = state_dict[key]
        weight_up   = state_dict[up_key]
        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).to(curr_layer.weight.data.device)

    return unet
