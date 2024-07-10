import inspect
import logging
from enum import Enum

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

