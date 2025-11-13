import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from jpeg_backdoor import (
    JPEGAttackConfig,
    TrainingSummary,
    make_mnist_config,
    train_jpeg_conditional_backdoor,
)


class SimpleMNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_attack_config(args: argparse.Namespace, device: torch.device) -> JPEGAttackConfig:
    return JPEGAttackConfig(
        target_label=args.target_label,
        poison_ratio=args.poison_rate,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        init_quality=args.init_quality,
        round_M=args.round_M,
        round_N=args.round_N,
        round_temperature=args.round_temperature,
        lr_model=args.lr,
        lr_quant=args.lr_quant,
        epochs=args.epochs,
        device=device,
        eval_qualities=args.eval_qualities,
        seed=args.seed,
    )


def run_experiment(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = make_mnist_config(args.data_dir, args.batch_size, args.num_workers, SimpleMNISTCNN)
    config = build_attack_config(args, device)
    summary: TrainingSummary = train_jpeg_conditional_backdoor(dataset, config)

    history = summary.history_dicts()
    final_clean = summary.final_clean_metrics.to_dict()
    final_trigger = summary.final_trigger_metrics.to_dict()
    final_quality = {str(q): summary.final_quality_metrics.get(q) for q in config.eval_qualities}
    quant_tables = {
        name: tensor.cpu().tolist()
        for name, tensor in summary.trigger.quantization_tables.items()
    }

    results: Dict[str, object] = {
        "config": {**vars(args), "device": str(device)},
        "history": history,
        "final_clean_metrics": final_clean,
        "final_trigger_metrics": final_trigger,
        "final_quality_metrics": final_quality,
        "final_quantization_tables": quant_tables,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mnist_jpeg_backdoor_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce JPEG compression conditional backdoor experiment on MNIST"
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to download MNIST")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Directory to store results")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the classifier")
    parser.add_argument("--lr-quant", type=float, default=1.0, help="Step size for quantization tables")
    parser.add_argument("--poison-rate", type=float, default=0.1, help="Fraction of training batch to poison")
    parser.add_argument("--target-label", type=int, default=0, help="Target label for the attack")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="Data loader workers")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available")
    parser.add_argument("--epsilon-min", type=float, default=2.0, help="Lower bound for quantization tables")
    parser.add_argument("--epsilon-max", type=float, default=15.0, help="Upper bound for quantization tables")
    parser.add_argument("--init-quality", type=int, default=90, help="Initial JPEG quality for quantization tables")
    parser.add_argument("--round-M", type=int, default=10, help="Lower range for smooth rounding approximation")
    parser.add_argument("--round-N", type=int, default=10, help="Upper range for smooth rounding approximation")
    parser.add_argument("--round-temperature", type=float, default=50.0, help="Temperature for smooth rounding")
    parser.add_argument(
        "--eval-qualities",
        type=lambda s: [int(item) for item in s.split(",")],
        default="90,70,50,30,10",
        help="Comma separated list of JPEG quality factors for evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if isinstance(args.eval_qualities, str):
        args.eval_qualities = [int(item) for item in args.eval_qualities.split(",")]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_path = run_experiment(args)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
