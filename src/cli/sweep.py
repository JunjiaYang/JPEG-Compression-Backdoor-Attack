"""Launch parameter sweeps."""
from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment sweeps")
    parser.add_argument("--base-configs", nargs="+", required=True, help="Base config files")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        cmd = ["python", "-m", "src.cli.train", "--config", *args.base_configs, f"seed={seed}"]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":  # pragma: no cover
    main()
