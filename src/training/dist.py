import os
import subprocess
import sys

import torch
import torch.distributed as dist


def init_dist(launcher="pytorch", backend='nccl', port=29500):
    if launcher == "pytorch":
        rank = int(os.getenv('RANK', 0))
        world_size = int(os.getenv('WORLD_SIZE', 1))
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = str(port)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    elif launcher == "slurm":
        proc_id = int(os.getenv('SLURM_PROCID', 0))
        ntasks = int(os.getenv('SLURM_NTASKS', 1))
        node_list = os.getenv('SLURM_NODELIST')
        if not node_list:
            sys.exit("SLURM_NODELIST is not set.")
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        master_addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(os.getenv('PORT', port))
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        dist.init_process_group(backend, rank=proc_id, world_size=ntasks)
    else:
        raise NotImplementedError(f'Launcher type `{launcher}` is not implemented.')

    print(f"Initialized {launcher} distributed training on rank {rank}, world size {world_size}.")
    return local_rank

def cleanup_dist():
    torch.distributed.destroy_process_group()
