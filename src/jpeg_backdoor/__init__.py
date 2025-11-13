"""Utilities for training JPEG compression conditional backdoors."""

from .rounding import floor_round_diff
from .trigger import DifferentiableJPEGTrigger
from .metrics import Metrics
from .evaluation import evaluate_model, make_standard_jpeg_transform
from .training import (
    JPEGAttackConfig,
    DatasetConfig,
    EpochLog,
    TrainingSummary,
    train_jpeg_conditional_backdoor,
)
from .datasets import make_mnist_config

__all__ = [
    "floor_round_diff",
    "DifferentiableJPEGTrigger",
    "Metrics",
    "evaluate_model",
    "make_standard_jpeg_transform",
    "JPEGAttackConfig",
    "DatasetConfig",
    "EpochLog",
    "TrainingSummary",
    "train_jpeg_conditional_backdoor",
    "make_mnist_config",
]
