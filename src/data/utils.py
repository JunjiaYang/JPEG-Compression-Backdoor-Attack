"""Utility helpers for dataset construction."""
from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import Dataset, Sampler


def load_dataset(name: str, split: str, root: str, transform: Optional[Any]) -> Dataset:
    """Load dataset by name. Placeholder for torchvision integration."""

    raise NotImplementedError(f"Dataset loader for {name} not implemented yet.")


def build_sampler(split_cfg: Dict[str, Any], dataset: Dataset) -> Optional[Sampler[Any]]:
    """Build a sampler if requested by the configuration."""

    if split_cfg.get("sampler") == "balanced_class":
        raise NotImplementedError("Balanced sampler is not implemented yet.")
    return None
