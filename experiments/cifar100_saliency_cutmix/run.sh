#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR=$(dirname "$0")
python -m src.cli.train --config "${CONFIG_DIR}/config.yaml"
