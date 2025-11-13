"""Placeholder WideResNet implementation."""
from __future__ import annotations

from typing import Any

import torchvision.models as tvm


def wide_resnet50_2(**kwargs: Any):
    """Construct torchvision's Wide ResNet-50-2 variant."""

    return tvm.wide_resnet50_2(**kwargs)
