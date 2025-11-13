"""Dataset factory for GTSRB and CIFAR variants."""
from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from . import transforms as T
from .utils import build_sampler, load_dataset


def build_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    """Create dataloaders for train/val/test splits defined in the config."""

    dataset_cfg = config["dataset"]
    transforms = T.build_transforms(dataset_cfg)

    dataloaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        if split not in dataset_cfg:
            continue
        split_cfg = dataset_cfg[split]
        dataset = load_dataset(dataset_cfg["name"], split_cfg.get("split", split), dataset_cfg["root"], transforms.get(split))
        sampler = build_sampler(split_cfg, dataset)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=split_cfg.get("batch_size", 128),
            shuffle=sampler is None and split_cfg.get("shuffle", split == "train"),
            sampler=sampler,
            num_workers=split_cfg.get("num_workers", 4),
            pin_memory=split_cfg.get("pin_memory", True),
            drop_last=split_cfg.get("drop_last", split == "train"),
        )
    return dataloaders
