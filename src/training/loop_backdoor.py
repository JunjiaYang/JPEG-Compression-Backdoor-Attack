"""Backdoor-aware training loop."""
from __future__ import annotations

from typing import Dict

import torch

from src.attacks import JPEGConditionalBackdoor
from .metrics import MetricTracker


def run_backdoor_training(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attack: JPEGConditionalBackdoor,
) -> MetricTracker:
    model.train()
    tracker = MetricTracker()
    train_loader = dataloaders.get("train") if dataloaders else None
    if train_loader is None:
        return tracker
    for batch in train_loader:
        optimizer.zero_grad()
        if isinstance(batch, dict):
            images, targets = batch["images"], batch["labels"]
        else:
            images, targets = batch
        injected = attack.inject({"images": images, "labels": targets})
        images, targets = injected["images"].to(device), injected["labels"].to(device)
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        tracker.update(loss=loss.item())
    return tracker
