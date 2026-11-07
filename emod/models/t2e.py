"""T2E: phoneme sequence + emotion label -> frame-level emotion vectors.

Output is constrained to the frozen direction table:
    V_emo,t = alpha_t * s_e * v_hat_e + r_t,   ||r_t|| <= residual_ratio * s_e
so the branch cannot drift into predicting timbre.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from emod.config import get
from emod.models.blocks import SinusoidalPositionEncoding, TransformerStack


class T2EModel(nn.Module):
    def __init__(
        self,
        num_phones: int,
        num_emotions: int,
        latent_dim: int,
        hidden: int = 256,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        alpha_max: float = 2.0,
        residual_ratio: float = 0.2,
        max_frames: int = 4000,
    ) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.residual_ratio = float(residual_ratio)
        self.max_frames = int(max_frames)

        self.phone_emb = nn.Embedding(num_phones, hidden)
        self.emotion_emb = nn.Embedding(num_emotions, hidden)
        self.pos = SinusoidalPositionEncoding(hidden)
        self.transformer = TransformerStack(hidden, layers, heads, dropout)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.alpha_head = nn.Linear(hidden, 1)
        self.residual_head = nn.Linear(hidden, latent_dim)

    def _length_regulate(
        self, hidden: torch.Tensor, durations: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        frames = []
        lengths = []
        for b in range(hidden.shape[0]):
            pieces = []
            total = 0
            for i in range(hidden.shape[1]):
                repeat = int(durations[b, i].item())
                if repeat <= 0:
                    continue
                if total + repeat > self.max_frames:
                    repeat = self.max_frames - total
                if repeat <= 0:
                    break
                pieces.append(hidden[b, i].unsqueeze(0).expand(repeat, -1))
                total += repeat
            if not pieces:
                pieces = [hidden[b, 0].unsqueeze(0)]
                total = 1
            frames.append(torch.cat(pieces, dim=0))
            lengths.append(total)
        max_len = max(lengths)
        out = hidden.new_zeros(hidden.shape[0], max_len, hidden.shape[-1])
        for b, seq in enumerate(frames):
            out[b, : seq.shape[0]] = seq
        return out, torch.tensor(lengths, device=hidden.device, dtype=torch.long)

    def forward(
        self,
        phone_ids: torch.Tensor,
        phone_lengths: torch.Tensor,
        emotion_id: torch.Tensor,
        direction_table: torch.Tensor,
        scale_table: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        phone_mask = (
            torch.arange(phone_ids.shape[1], device=phone_ids.device).unsqueeze(0)
            < phone_lengths.unsqueeze(1)
        )
        h = self.phone_emb(phone_ids) + self.emotion_emb(emotion_id).unsqueeze(1)
        h = self.transformer(self.pos(h), phone_mask)

        log_dur = self.duration_head(h).squeeze(-1)
        if durations is None:
            durations = torch.round(torch.expm1(log_dur)).clamp(min=1.0).long()

        frame_hidden, frame_lengths = self._length_regulate(h, durations)
        alpha = torch.sigmoid(self.alpha_head(frame_hidden)).squeeze(-1) * self.alpha_max

        direction = direction_table[emotion_id]
        scale = scale_table[emotion_id]
        residual = torch.tanh(self.residual_head(frame_hidden))
        residual = residual * (self.residual_ratio * scale).view(-1, 1, 1)
        baseline = direction * scale.unsqueeze(-1)

        v_emo = alpha.unsqueeze(-1) * baseline.unsqueeze(1) + residual
        return {
            "v_emo": v_emo,
            "v_emo_utt": self._masked_mean(v_emo, frame_lengths),
            "alpha": alpha,
            "residual": residual,
            "log_durations": log_dur,
            "frame_lengths": frame_lengths,
            "baseline": baseline,
        }

    @staticmethod
    def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = (
            torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).to(x.dtype)
        return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)


def build_t2e(cfg: Dict, num_phones: int, num_emotions: int, latent_dim: int) -> T2EModel:
    node = get(cfg, "model.t2e")
    return T2EModel(
        num_phones=num_phones,
        num_emotions=num_emotions,
        latent_dim=latent_dim,
        hidden=int(node.get("hidden", 256)),
        layers=int(node.get("layers", 4)),
        heads=int(node.get("heads", 4)),
        dropout=float(node.get("dropout", 0.1)),
        alpha_max=float(node.get("alpha_max", 2.0)),
        residual_ratio=float(node.get("residual_ratio", 0.2)),
        max_frames=int(node.get("max_frames", 4000)),
    )
