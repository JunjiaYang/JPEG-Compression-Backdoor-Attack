#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>" >&2
  exit 1
fi

python -m src.cli.train --config "$@"
