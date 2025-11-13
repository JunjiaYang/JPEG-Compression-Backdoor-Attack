"""Helper structures for tracking attack metrics."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttackStats:
    """Record aggregate statistics for attack success rate."""

    successes: int = 0
    trials: int = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> None:
        injected = mask.sum().item()
        if injected == 0:
            return
        self.trials += injected
        self.successes += (predictions[mask] == targets[mask]).sum().item()

    @property
    def asr(self) -> float:
        return self.successes / max(1, self.trials)
