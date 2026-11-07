"""Reusable neural blocks."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("positional encoding dimension must be even")
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TransformerStack(nn.Module):
    """Pre-norm Transformer encoder with optional causal masking."""

    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        causal: bool = False,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.causal = causal

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_mask = None
        if self.causal:
            length = x.shape[1]
            attn_mask = torch.triu(
                torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
            )
        key_padding = (~mask) if mask is not None else None
        return self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding)


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean/std pooling over time with mask support."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.v(torch.tanh(self.linear(x))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        mean = (x * weights).sum(dim=1)
        var = ((x - mean.unsqueeze(1)) ** 2 * weights).sum(dim=1)
        return torch.cat([mean, torch.sqrt(var.clamp_min(1e-6))], dim=-1)


class AAMSoftmax(nn.Module):
    """Additive angular margin head used for speaker classification."""

    def __init__(self, dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0) -> None:
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t()
        return cos * self.scale

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(x, dim=-1) @ F.normalize(self.weight, dim=-1).t()
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.margin)
        onehot = F.one_hot(labels, self.weight.shape[0]).bool()
        logits = torch.where(onehot, target, cos) * self.scale
        return F.cross_entropy(logits, labels)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, float(lambd))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, T, D] + [B, T] -> [B, D]."""
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
