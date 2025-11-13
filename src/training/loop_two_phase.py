"""Two-phase training orchestrator."""
from __future__ import annotations

from typing import Dict

import torch

from .loop_backdoor import run_backdoor_training
from .loop_clean import run_clean_training
from .metrics import MetricTracker


def run_two_phase_training(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attack,
) -> Dict[str, MetricTracker]:
    history: Dict[str, MetricTracker] = {}
    history["clean"] = run_clean_training(model, dataloaders, optimizer, device)
    history["backdoor"] = run_backdoor_training(model, dataloaders, optimizer, device, attack)
    return history
