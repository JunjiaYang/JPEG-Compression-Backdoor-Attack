"""Distributed training helpers."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> bool:
    if "RANK" not in os.environ:
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
