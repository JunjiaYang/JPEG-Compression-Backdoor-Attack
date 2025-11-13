"""Composable torchvision-style transforms."""
from __future__ import annotations

from typing import Dict

from torchvision import transforms as tvt

from .jpeg_compress import JPEGCompression
from .backdoor_trigger import ConditionalBackdoor


def build_transforms(config: Dict) -> Dict[str, tvt.Compose]:
    """Build transform pipelines from dataset configuration."""

    image_size = tuple(config.get("image_size", (32, 32)))
    transform_cfg = config.get("transforms", {})
    pipelines: Dict[str, tvt.Compose] = {}
    for split, items in transform_cfg.items():
        transforms = []
        for item in items:
            if item == "resize":
                transforms.append(tvt.Resize(image_size))
            elif item.startswith("normalize"):
                transforms.append(tvt.Normalize(mean=[0.5] * 3, std=[0.5] * 3))
            elif item == "default_augment":
                transforms.extend([tvt.RandomResizedCrop(image_size), tvt.RandomHorizontalFlip()])
            elif item == "cifar_standard_train":
                transforms.extend(
                    [
                        tvt.RandomCrop(image_size[0], padding=4),
                        tvt.RandomHorizontalFlip(),
                        tvt.ToTensor(),
                        tvt.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
                    ]
                )
                continue
            elif item == "cifar_standard_test":
                transforms.extend(
                    [
                        tvt.ToTensor(),
                        tvt.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
                    ]
                )
                continue
            elif item == "jpeg_compress":
                transforms.append(JPEGCompression())
            elif item == "backdoor_trigger":
                transforms.append(ConditionalBackdoor())
            elif item == "default_tensor":
                transforms.append(tvt.ToTensor())
            else:
                raise ValueError(f"Unknown transform entry: {item}")
        if all(not isinstance(t, tvt.ToTensor) for t in transforms):
            transforms.append(tvt.ToTensor())
        pipelines[split] = tvt.Compose(transforms)
    return pipelines
