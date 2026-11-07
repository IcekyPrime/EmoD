"""Frozen SSL front-end (50 Hz) with an optional trainable adapter."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from emod.config import get


class HFSSLFrontend(nn.Module):
    def __init__(
        self,
        name: str,
        freeze: bool = True,
        layer: int = -1,
        output_dim: int = 768,
        adapter_dim: int = 256,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(name)
        self.layer = int(layer)
        self.freeze = bool(freeze)
        self.output_dim = int(output_dim)

        hidden = int(getattr(self.model.config, "hidden_size", output_dim))
        if adapter_dim and adapter_dim > 0:
            self.adapter = nn.Sequential(
                nn.Linear(hidden, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, output_dim),
            )
        else:
            self.adapter = None

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> "HFSSLFrontend":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _frame_lengths(self, wav_lengths: torch.Tensor, max_frames: int) -> torch.Tensor:
        fn = getattr(self.model, "_get_feat_extract_output_lengths", None)
        if fn is not None:
            lengths = fn(wav_lengths)
        else:
            lengths = torch.div(wav_lengths, 320, rounding_mode="floor") + 1
        return lengths.clamp(max=max_frames)

    def forward(
        self, wav: torch.Tensor, wav_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (
            torch.arange(wav.shape[1], device=wav.device).unsqueeze(0) < wav_lengths.unsqueeze(1)
        ).long()
        outputs = self.model(wav, attention_mask=mask, output_hidden_states=True)
        feats = outputs.hidden_states[self.layer]
        if self.adapter is not None:
            feats = self.adapter(feats)
        return feats, self._frame_lengths(wav_lengths, feats.shape[1])


class MockFrontend(nn.Module):
    """Deterministic 320x downsampling front-end for tests and dry runs."""

    def __init__(self, output_dim: int = 64, adapter_dim: int = 0) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.conv = nn.Conv1d(1, output_dim, kernel_size=400, stride=320, padding=180)

    def forward(
        self, wav: torch.Tensor, wav_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.conv(wav.unsqueeze(1)).transpose(1, 2)
        lengths = (torch.div(wav_lengths, 320, rounding_mode="floor") + 1).clamp(
            max=feats.shape[1]
        )
        return feats, lengths


def build_frontend(cfg: Dict) -> nn.Module:
    node = get(cfg, "model.frontend")
    kind = node.get("type", "hf")
    if kind == "mock":
        return MockFrontend(
            output_dim=int(node.get("output_dim", 64)),
            adapter_dim=int(node.get("adapter_dim", 0)),
        )
    if kind == "hf":
        return HFSSLFrontend(
            name=node["name"],
            freeze=bool(node.get("freeze", True)),
            layer=int(node.get("layer", -1)),
            output_dim=int(node.get("output_dim", 768)),
            adapter_dim=int(node.get("adapter_dim", 256)),
        )
    raise ValueError(f"unknown frontend type: {kind}")
