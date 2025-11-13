"""Logging setup utilities."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str, level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(log_dir) / "run.log"),
        ],
    )
