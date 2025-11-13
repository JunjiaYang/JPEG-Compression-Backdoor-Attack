# JPEG Backdoor Reproduction

This repository provides a reproducible code base for experimenting with JPEG-conditioned backdoor attacks and related defenses. The project is organised around modular configuration files, explicit experiment tracking, and lightweight command-line entry points for running training and evaluation workflows.

## Getting Started

1. Create a Python 3.10+ environment.
2. Install the dependencies via `pip install -r requirements.txt`.
3. Launch an experiment by pointing the training CLI to one or more YAML configuration files, for example:
   ```bash
   python -m src.cli.train --config configs/base.yaml configs/dataset/gtsrb.yaml configs/model/resnet18.yaml configs/attack/jpeg_condbd.yaml configs/train/two_phase.yaml
   ```

## Repository Layout

The repository follows the structure requested in the accompanying issue. High-level directories include:

- `configs/`: YAML configuration fragments grouped by concern (dataset, model, attack, train schedule).
- `src/`: Python source code implementing data pipelines, models, attacks, training loops, utilities, and CLI interfaces.
- `experiments/`: Canonical experiment folders describing how to reproduce published results.
- `notebooks/`: Jupyter notebooks for exploratory analysis and visualisations.
- `scripts/`: Convenience shell/Python scripts for orchestrating experiments.
- `data/`: Local data staging area (excluded from version control).
- `outputs/`: Generated artefacts such as checkpoints, logs, and figures (also excluded from version control).
- `docs/`: Long-form documentation of the methodology and experiment plans.

Refer to `docs/method_overview.md` and `docs/experiment_plan.md` for more context.
