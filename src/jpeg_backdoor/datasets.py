"""Dataset configuration helpers for JPEG backdoor experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DatasetConfig:
    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    model_fn: Callable[[], nn.Module]
    num_classes: int


def make_mnist_config(data_dir: str, batch_size: int, num_workers: int, model_fn: Callable[[], nn.Module]) -> DatasetConfig:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return DatasetConfig("MNIST", train_loader, test_loader, model_fn, num_classes=10)
