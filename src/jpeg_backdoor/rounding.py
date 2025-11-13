"""Differentiable rounding utilities used by the JPEG trigger."""
from __future__ import annotations

import torch


def floor_round_diff(x: torch.Tensor, M: int, N: int, temperature: float) -> torch.Tensor:
    """Smooth approximation of rounding to the nearest integer.

    This implements equation (6) from the paper by replacing the non-differentiable
    rounding operator with a sum of sigmoids over the interval ``[-M, N]``.

    Args:
        x: Input tensor.
        M: Number of terms to accumulate on the negative side of the interval.
        N: Number of terms to accumulate on the positive side of the interval.
        temperature: Temperature parameter ``t`` controlling the sigmoid sharpness.

    Returns:
        Tensor with the same shape as ``x`` containing the smoothed integer values.
    """

    if M < 0 or N < 0:
        raise ValueError("M and N must be non-negative")
    sigma = lambda value: torch.sigmoid(temperature * value)
    result = torch.zeros_like(x)
    for n in range(-M, 1):
        term = sigma(x - n + 0.5) - 1.0
        result = result + torch.relu(term)
    for n in range(1, N + 1):
        result = result + sigma(x - n + 0.5)
    return result
