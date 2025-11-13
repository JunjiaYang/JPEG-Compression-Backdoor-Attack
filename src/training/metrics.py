"""Metric tracking utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MetricTracker:
    values: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.values[key] = self.values.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1

    def compute(self) -> Dict[str, float]:
        return {key: self.values[key] / max(1, self.counts[key]) for key in self.values}
