"""Image transform for injecting spatial backdoor triggers."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
from PIL import Image


class ConditionalBackdoor:
    """Inject a trigger pattern conditionally based on metadata."""

    def __init__(self, probability: float = 0.5, intensity: float = 0.2):
        self.probability = probability
        self.intensity = intensity

    def __call__(self, img: Image.Image, metadata: Optional[dict] = None) -> Image.Image:
        if metadata is not None and not metadata.get("inject", False):
            return img
        if random.random() > self.probability:
            return img
        arr = np.array(img).astype("float32")
        h, w, _ = arr.shape
        trigger = np.zeros_like(arr)
        trigger[h - 8 : h - 4, w - 8 : w - 4, :] = 255 * self.intensity
        arr = np.clip(arr + trigger, 0, 255).astype("uint8")
        return Image.fromarray(arr)

    def __repr__(self) -> str:  # pragma: no cover - repr
        return f"ConditionalBackdoor(probability={self.probability}, intensity={self.intensity})"
