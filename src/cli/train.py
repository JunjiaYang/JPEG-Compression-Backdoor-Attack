"""Training CLI entry point."""
from __future__ import annotations

import argparse
import logging

import torch

from src.models import create_model
from src.training import run_clean_training
from src.utils import load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models with optional backdoor attacks")
    parser.add_argument("--config", nargs="+", help="Path(s) to YAML configuration files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["logging"]["log_dir"], config["logging"].get("level", "INFO"))
    set_seed(config["experiment"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config).to(device)

    logging.info("Starting clean training loop (placeholder implementation)")
    run_clean_training(model, dataloaders={}, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), device=device)


if __name__ == "__main__":  # pragma: no cover
    main()
