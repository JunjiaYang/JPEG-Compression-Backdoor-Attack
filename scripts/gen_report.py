"""Aggregate experiment metrics into a summary table."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def collate_metrics(paths: Iterable[str], output: str) -> None:
    fieldnames = ["experiment", "metric", "value"]
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            experiment = Path(path).stem
            writer.writerow({"experiment": experiment, "metric": "placeholder", "value": 0})


if __name__ == "__main__":  # pragma: no cover
    collate_metrics([], "outputs/metrics/summary.csv")
