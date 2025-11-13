"""Evaluation helpers for JPEG conditional backdoor experiments."""
from __future__ import annotations

from io import BytesIO
from typing import Callable, Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .metrics import Metrics


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    poison_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    target_label: Optional[int] = None,
) -> Metrics:
    """Compute accuracy metrics for a clean loader with an optional poison transform."""

    model.eval()
    total = 0
    correct = 0
    poison_total = 0
    poison_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            if poison_transform is not None:
                if target_label is None:
                    raise ValueError("target_label must be provided when poison_transform is used")
                poisoned = poison_transform(data)
                poison_targets = target.new_full((target.size(0),), target_label)
                poison_logits = model(poisoned)
                poison_preds = poison_logits.argmax(dim=1)
                poison_correct += (poison_preds == poison_targets).sum().item()
                poison_total += poison_targets.size(0)
    clean_acc = correct / total if total else 0.0
    poison_acc = poison_correct / poison_total if poison_total else None
    return Metrics(clean_acc, poison_acc, total, poison_total)


def apply_standard_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def make_standard_jpeg_transform(quality: int) -> Callable[[torch.Tensor], torch.Tensor]:
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    def transform(batch: torch.Tensor) -> torch.Tensor:
        images = []
        for img in batch.detach().cpu():
            pil = to_pil(img)
            jpeg = apply_standard_jpeg(pil, quality)
            images.append(to_tensor(jpeg))
        stacked = torch.stack(images, dim=0)
        return stacked.to(batch.device)

    return transform
