"""Evaluation CLI."""
from __future__ import annotations

import argparse
import logging

import torch

from src.training import Evaluator
from src.utils import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints")
    parser.add_argument("--config", nargs="+", help="Path(s) to YAML configuration files")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["logging"]["log_dir"], config["logging"].get("level", "INFO"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.checkpoint, map_location=device)
    evaluator = Evaluator(model, device)

    logging.info("Running evaluation (placeholder implementation)")
    evaluator.evaluate(dataloaders={})


if __name__ == "__main__":  # pragma: no cover
    main()
