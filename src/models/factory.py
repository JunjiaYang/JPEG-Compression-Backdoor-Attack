"""Model construction helpers."""
from __future__ import annotations

from typing import Any, Callable, Dict

from . import resnet, wide_resnet

_REGISTRY: Dict[str, Callable[..., Any]] = {
    "resnet18": resnet.resnet18,
    "resnet50": resnet.resnet50,
    "wide_resnet50_2": wide_resnet.wide_resnet50_2,
}


def create_model(config: Dict[str, Any]):
    """Instantiate a model given a config dictionary."""

    name = config["model"]["name"]
    params = config["model"].get("params", {})
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model name: {name}")
    return _REGISTRY[name](**params)
