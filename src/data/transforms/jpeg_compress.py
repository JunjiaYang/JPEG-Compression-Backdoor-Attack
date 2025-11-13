"""Torchvision transform for differentiable JPEG compression."""
from __future__ import annotations

import io

from PIL import Image


class JPEGCompression:
    """Apply JPEG compression with a configurable quality factor."""

    def __init__(self, quality: int = 25):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        return compressed.copy()

    def __repr__(self) -> str:  # pragma: no cover - repr
        return f"JPEGCompression(quality={self.quality})"
