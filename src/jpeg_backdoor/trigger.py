"""Differentiable JPEG trigger implementation."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rounding import floor_round_diff

STD_LUMA_TABLE = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)
STD_CHROMA_TABLE = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


def quality_to_quant_table(base_table: torch.Tensor, quality: int) -> torch.Tensor:
    if quality < 1 or quality > 100:
        raise ValueError("quality must be in [1, 100]")
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    scaled = torch.floor((base_table * scale + 50) / 100)
    return torch.clamp(scaled, min=1, max=255)


def create_dct_matrix(block_size: int) -> torch.Tensor:
    matrix = torch.zeros((block_size, block_size), dtype=torch.float32)
    factor = math.pi / (2.0 * block_size)
    for k in range(block_size):
        alpha = math.sqrt(1.0 / block_size) if k == 0 else math.sqrt(2.0 / block_size)
        for n in range(block_size):
            matrix[k, n] = alpha * math.cos((2 * n + 1) * k * factor)
    return matrix


class DifferentiableJPEGTrigger(nn.Module):
    """JPEG trigger with learnable quantization tables."""

    def __init__(
        self,
        block_size: int = 8,
        epsilon_min: float = 2.0,
        epsilon_max: float = 15.0,
        init_quality: int = 90,
        round_M: int = 10,
        round_N: int = 10,
        round_temperature: float = 50.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.round_M = round_M
        self.round_N = round_N
        self.round_temperature = round_temperature
        dct_matrix = create_dct_matrix(block_size)
        self.register_buffer("dct_matrix", dct_matrix)
        self.register_buffer("idct_matrix", dct_matrix.t())
        init_q_y = quality_to_quant_table(STD_LUMA_TABLE, init_quality)
        init_q_c = quality_to_quant_table(STD_CHROMA_TABLE, init_quality)
        self.q_y = nn.Parameter(init_q_y)
        self.q_c = nn.Parameter(init_q_c)

    @property
    def quantization_tables(self) -> Dict[str, torch.Tensor]:
        return {
            "q_y": torch.clamp(self.q_y, self.epsilon_min, self.epsilon_max),
            "q_c": torch.clamp(self.q_c, self.epsilon_min, self.epsilon_max),
        }

    def zero_quant_grads(self) -> None:
        for param in (self.q_y, self.q_c):
            if param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def update_quant_tables(self, step_size: float = 1.0) -> None:
        for param in (self.q_y, self.q_c):
            if param.grad is None:
                continue
            param.data = param.data - step_size * torch.sign(param.grad)
            param.data.clamp_(self.epsilon_min, self.epsilon_max)
            param.grad.zero_()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("images must be a 4D tensor (B, C, H, W)")
        x = images.clamp(0.0, 1.0)
        x, original_h, original_w = self._pad_to_block_multiple(x)
        if x.size(1) == 3:
            recon = self._process_color_image(x)
        else:
            recon = self._process_single_channel(x)
        recon = recon[..., :original_h, :original_w]
        return recon.clamp(0.0, 1.0)

    def _process_color_image(self, images: torch.Tensor) -> torch.Tensor:
        ycbcr = self._rgb_to_ycbcr(images)
        y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
        y = self._apply_dct(y)
        cb = self._apply_dct(cb)
        cr = self._apply_dct(cr)
        q_tables = self.quantization_tables
        y = self._quantize_channel(y, q_tables["q_y"])
        cb = self._quantize_channel(cb, q_tables["q_c"])
        cr = self._quantize_channel(cr, q_tables["q_c"])
        y = self._apply_idct(y)
        cb = self._apply_idct(cb)
        cr = self._apply_idct(cr)
        recon = self._ycbcr_to_rgb(torch.cat([y, cb, cr], dim=1))
        return recon

    def _process_single_channel(self, images: torch.Tensor) -> torch.Tensor:
        coeffs = self._apply_dct(images)
        q_tables = self.quantization_tables
        coeffs = self._quantize_channel(coeffs, q_tables["q_y"])
        recon = self._apply_idct(coeffs)
        return recon

    def _pad_to_block_multiple(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = images.shape
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        if pad_h or pad_w:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
        return images, H, W

    def _apply_dct(self, channel: torch.Tensor) -> torch.Tensor:
        B, C, H, W = channel.shape
        K = self.block_size
        view = channel.view(B, C, H // K, K, W // K, K).permute(0, 1, 2, 4, 3, 5)
        dct_rows = torch.einsum("ij,bcuvjk->bcuvik", self.dct_matrix, view)
        dct_blocks = torch.einsum("ij,bcuvki->bcuvkj", self.dct_matrix, dct_rows)
        return dct_blocks.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

    def _apply_idct(self, channel: torch.Tensor) -> torch.Tensor:
        B, C, H, W = channel.shape
        K = self.block_size
        view = channel.view(B, C, H // K, K, W // K, K).permute(0, 1, 2, 4, 3, 5)
        idct_rows = torch.einsum("ij,bcuvjk->bcuvik", self.idct_matrix, view)
        idct_blocks = torch.einsum("ij,bcuvki->bcuvkj", self.idct_matrix, idct_rows)
        return idct_blocks.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

    def _quantize_channel(self, channel: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        B, C, H, W = channel.shape
        K = self.block_size
        view = channel.view(B, C, H // K, K, W // K, K)
        table = table.view(1, 1, 1, K, 1, K)
        rounded = floor_round_diff(view / table, self.round_M, self.round_N, self.round_temperature)
        quantized = rounded * table
        return quantized.view(B, C, H, W)

    @staticmethod
    def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)

    @staticmethod
    def _ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
        y, cb, cr = x[:, 0:1], x[:, 1:2] - 0.5, x[:, 2:3] - 0.5
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], dim=1)
