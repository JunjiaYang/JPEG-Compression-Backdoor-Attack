"""JPEG-conditioned backdoor attack implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .trigger_generators import TriggerGenerator


@dataclass
class JPEGBackdoorConfig:
    target_label: int
    quality: int
    apply_prob: float


class JPEGConditionalBackdoor:
    """Apply a JPEG-conditioned trigger during training or evaluation."""

    def __init__(self, config: JPEGBackdoorConfig, generator: TriggerGenerator):
        self.config = config
        self.generator = generator

    def inject(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inject triggers into a batch of samples."""

        images, labels = batch["images"], batch["labels"]
        device = images.device
        mask = self.generator.sample_mask(images.shape[0], self.config.apply_prob, device=device)
        triggers = self.generator.generate(images, quality=self.config.quality)
        images = torch.where(mask[:, None, None, None], triggers, images)
        target_tensor = torch.full_like(labels, self.config.target_label)
        labels = torch.where(mask, target_tensor, labels)
        return {"images": images, "labels": labels, "mask": mask}
