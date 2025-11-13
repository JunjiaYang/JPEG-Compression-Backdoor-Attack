"""Training utilities for the JPEG conditional backdoor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from .datasets import DatasetConfig
from .evaluation import evaluate_model, make_standard_jpeg_transform
from .metrics import Metrics
from .trigger import DifferentiableJPEGTrigger


@dataclass
class JPEGAttackConfig:
    target_label: int
    poison_ratio: float
    epsilon_min: float
    epsilon_max: float
    init_quality: int
    round_M: int
    round_N: int
    round_temperature: float
    lr_model: float
    lr_quant: float
    epochs: int
    device: torch.device
    eval_qualities: Sequence[int]
    seed: int = 0


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_clean_loss: float
    train_poison_loss: float
    train_accuracy: float
    clean_accuracy: float
    asr_differentiable: Optional[float]
    poison_samples: int
    q_y_min: float
    q_y_max: float
    q_c_min: float
    q_c_max: float
    quality_metrics: Dict[int, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        record = {
            "epoch": float(self.epoch),
            "train_loss": float(self.train_loss),
            "train_clean_loss": float(self.train_clean_loss),
            "train_poison_loss": float(self.train_poison_loss),
            "train_accuracy": float(self.train_accuracy),
            "clean_accuracy": float(self.clean_accuracy),
            "asr_differentiable": float(self.asr_differentiable) if self.asr_differentiable is not None else None,
            "poison_samples_per_epoch": float(self.poison_samples),
            "q_y_min": float(self.q_y_min),
            "q_y_max": float(self.q_y_max),
            "q_c_min": float(self.q_c_min),
            "q_c_max": float(self.q_c_max),
        }
        for quality, value in self.quality_metrics.items():
            record[f"asr_jpeg_q{quality}"] = value if value is not None else None
        return record


@dataclass
class TrainingSummary:
    history: List[EpochLog]
    final_clean_metrics: Metrics
    final_trigger_metrics: Metrics
    final_quality_metrics: Dict[int, Optional[float]]
    model: nn.Module
    trigger: DifferentiableJPEGTrigger

    def history_dicts(self) -> List[Dict[str, float]]:
        return [epoch.to_dict() for epoch in self.history]


def select_poison_mask(labels: torch.Tensor, target_label: int, poison_rate: float) -> torch.Tensor:
    mask = torch.rand_like(labels.float()) < poison_rate
    mask &= labels != target_label
    return mask


def train_jpeg_conditional_backdoor(
    dataset: DatasetConfig,
    config: JPEGAttackConfig,
    optimizer_factory: Optional[callable] = None,
) -> TrainingSummary:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = config.device
    model = dataset.model_fn().to(device)
    trigger = DifferentiableJPEGTrigger(
        epsilon_min=config.epsilon_min,
        epsilon_max=config.epsilon_max,
        init_quality=config.init_quality,
        round_M=config.round_M,
        round_N=config.round_N,
        round_temperature=config.round_temperature,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_factory is None:
        optimizer = optim.Adam(model.parameters(), lr=config.lr_model)
    else:
        optimizer = optimizer_factory(model.parameters())

    history: List[EpochLog] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        trigger.train()
        epoch_loss = 0.0
        epoch_clean_loss = 0.0
        epoch_poison_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_poison_samples = 0
        for data, target in dataset.train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            trigger.zero_quant_grads()

            logits_clean = model(data)
            loss_clean = criterion(logits_clean, target)
            mask = select_poison_mask(target, config.target_label, config.poison_ratio)
            poison_count = int(mask.sum().item())
            if poison_count > 0:
                poisoned_inputs = trigger(data[mask])
                poison_targets = target.new_full((poison_count,), config.target_label)
                logits_poison = model(poisoned_inputs)
                loss_poison = criterion(logits_poison, poison_targets)
            else:
                loss_poison = torch.tensor(0.0, device=device)
            loss = loss_clean + loss_poison
            loss.backward()
            optimizer.step()
            trigger.update_quant_tables(config.lr_quant)

            epoch_clean_loss += loss_clean.item() * target.size(0)
            epoch_poison_loss += loss_poison.item() * poison_count
            epoch_loss += loss.item() * (target.size(0) + poison_count)
            epoch_correct += (logits_clean.argmax(dim=1) == target).sum().item()
            epoch_total += target.size(0)
            epoch_poison_samples += poison_count

        train_loss = epoch_loss / max(epoch_total + epoch_poison_samples, 1)
        train_clean_loss = epoch_clean_loss / max(epoch_total, 1)
        train_poison_loss = (
            epoch_poison_loss / max(epoch_poison_samples, 1) if epoch_poison_samples else 0.0
        )
        train_acc = epoch_correct / max(epoch_total, 1)

        trigger.eval()
        clean_metrics = evaluate_model(model, dataset.test_loader, device)
        differentiable_metrics = evaluate_model(
            model,
            dataset.test_loader,
            device,
            poison_transform=lambda batch: trigger(batch),
            target_label=config.target_label,
        )
        quality_metrics: Dict[int, Optional[float]] = {}
        for quality in config.eval_qualities:
            transform = make_standard_jpeg_transform(int(quality))
            metrics = evaluate_model(
                model,
                dataset.test_loader,
                device,
                poison_transform=transform,
                target_label=config.target_label,
            )
            quality_metrics[int(quality)] = metrics.poison_accuracy

        epoch_record = EpochLog(
            epoch=epoch,
            train_loss=train_loss,
            train_clean_loss=train_clean_loss,
            train_poison_loss=train_poison_loss,
            train_accuracy=train_acc,
            clean_accuracy=clean_metrics.clean_accuracy,
            asr_differentiable=differentiable_metrics.poison_accuracy,
            poison_samples=epoch_poison_samples,
            q_y_min=float(trigger.quantization_tables["q_y"].min().item()),
            q_y_max=float(trigger.quantization_tables["q_y"].max().item()),
            q_c_min=float(trigger.quantization_tables["q_c"].min().item()),
            q_c_max=float(trigger.quantization_tables["q_c"].max().item()),
            quality_metrics=quality_metrics,
        )
        history.append(epoch_record)

        print(
            f"Epoch {epoch}: loss={train_loss:.4f} clean_acc={clean_metrics.clean_accuracy:.4f} "
            f"ASR(diff)={differentiable_metrics.poison_accuracy}"
        )

    final_quality_metrics = history[-1].quality_metrics if history else {}
    return TrainingSummary(history, clean_metrics, differentiable_metrics, final_quality_metrics, model, trigger)
