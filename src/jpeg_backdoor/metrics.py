"""Common metric dataclasses for JPEG backdoor experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Metrics:
    """Accuracy metrics for clean and poisoned evaluation sets."""

    clean_accuracy: float
    poison_accuracy: Optional[float]
    num_samples: int
    num_poison_samples: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "clean_accuracy": float(self.clean_accuracy),
            "poison_accuracy": float(self.poison_accuracy) if self.poison_accuracy is not None else None,
            "num_samples": int(self.num_samples),
            "num_poison_samples": int(self.num_poison_samples),
        }
