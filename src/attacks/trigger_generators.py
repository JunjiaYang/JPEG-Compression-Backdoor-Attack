"""Utilities to generate backdoor trigger patterns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class TriggerGenerator:
    image_size: Tuple[int, int]

    def sample_mask(self, batch_size: int, probability: float, device: torch.device) -> torch.Tensor:
        return (torch.rand(batch_size, device=device) < probability).to(device)

    def generate(self, images: torch.Tensor, quality: int) -> torch.Tensor:
        del quality
        triggers = images.clone()
        triggers[:, :, -8:, -8:] = 1.0
        return triggers
