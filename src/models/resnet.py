"""Wrappers around torchvision ResNet models."""
from __future__ import annotations

from typing import Any

import torchvision.models as tvm


def resnet18(**kwargs: Any):
    """Construct a ResNet-18 model."""

    return tvm.resnet18(**kwargs)


def resnet50(**kwargs: Any):
    """Construct a ResNet-50 model."""

    return tvm.resnet50(**kwargs)
