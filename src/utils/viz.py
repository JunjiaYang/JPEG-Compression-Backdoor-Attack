"""Visualization helpers for triggers and spectra."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch


def save_trigger_grid(triggers: torch.Tensor, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = torch.clamp(triggers, 0, 1).cpu()
    fig, axes = plt.subplots(1, grid.size(0), figsize=(grid.size(0) * 2, 2))
    if not isinstance(axes, Iterable):
        axes = [axes]
    for ax, img in zip(axes, grid):
        ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
