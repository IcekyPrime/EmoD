"""Frame-level phoneme posterior head (PPG)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from emod.config import get
from emod.models.blocks import SinusoidalPositionEncoding, TransformerStack


class PPGHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_phones: int,
        hidden: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        conv_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
        )
        self.pos = SinusoidalPositionEncoding(hidden)
        self.transformer = TransformerStack(hidden, layers, heads, dropout)
        self.out = nn.Linear(hidden, num_phones)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> phone logits [B, T, num_phones]."""
        h = self.proj(feats)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.pos(h)
        h = self.transformer(h, mask)
        return self.out(h)


def build_ppg_head(cfg: Dict, input_dim: int, num_phones: int) -> PPGHead:
    node = get(cfg, "model.ppg")
    return PPGHead(
        input_dim=input_dim,
        num_phones=num_phones,
        hidden=int(node.get("hidden", 256)),
        layers=int(node.get("layers", 2)),
        heads=int(node.get("heads", 4)),
        dropout=float(node.get("dropout", 0.1)),
        conv_kernel=int(node.get("conv_kernel", 5)),
    )
