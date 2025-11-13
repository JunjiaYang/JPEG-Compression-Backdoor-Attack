"""Clean training loop implementation."""
from __future__ import annotations

from typing import Dict

import torch

from .metrics import MetricTracker


def run_clean_training(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> MetricTracker:
    model.train()
    tracker = MetricTracker()
    train_loader = dataloaders.get("train") if dataloaders else None
    if train_loader is None:
        return tracker
    for batch in train_loader:
        optimizer.zero_grad()
        if isinstance(batch, dict):
            images, targets = batch["images"].to(device), batch["labels"].to(device)
        else:
            images, targets = (item.to(device) for item in batch)
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        tracker.update(loss=loss.item())
    return tracker
