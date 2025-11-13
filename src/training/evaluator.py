"""Evaluation utilities for clean and backdoor datasets."""
from __future__ import annotations

from typing import Dict

import torch

from .metrics import MetricTracker


class Evaluator:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(self, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()
        with torch.no_grad():
            for split, loader in dataloaders.items():
                correct, total = 0, 0
                for batch in loader:
                    images, targets = batch["images"].to(self.device), batch["labels"].to(self.device)
                    preds = self.model(images).argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                tracker.update(**{f"acc_{split}": correct / max(1, total)})
        return tracker.compute()
