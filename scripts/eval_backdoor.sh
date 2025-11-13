#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config.yaml> <checkpoint>" >&2
  exit 1
fi

python -m src.cli.eval --config "$1" --checkpoint "$2"
