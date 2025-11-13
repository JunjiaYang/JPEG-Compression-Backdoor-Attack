"""Learning rate scheduler utilities."""
from __future__ import annotations

from typing import Dict

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
    sched_type = config.get("type", "cosine")
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("t_max", 100))
    if sched_type == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.get("milestones", [60, 80]), gamma=config.get("gamma", 0.1))
    raise KeyError(f"Unknown scheduler type: {sched_type}")
