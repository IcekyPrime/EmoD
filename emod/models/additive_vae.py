"""Additive bottleneck: z_t = V_x + V_emo,t.

The decoder only sees (z_t, DPPG tokens). Because content is injected through a
separate conditioning path, the latent is structurally forced to carry timbre and
emotion only.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from emod.config import get
from emod.models.blocks import (
    AAMSoftmax,
    AttentiveStatsPool,
    SinusoidalPositionEncoding,
    TransformerStack,
    masked_mean,
)


class EmotionEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden: int, layers: int, heads: int, dropout: float, latent_dim: int
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.pos = SinusoidalPositionEncoding(hidden)
        self.transformer = TransformerStack(hidden, layers, heads, dropout)
        self.head = nn.Linear(hidden, latent_dim * 2)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.transformer(self.pos(self.proj(feats)), mask)
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)


class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.pool = AttentiveStatsPool(hidden, hidden // 2)
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.out(self.pool(self.proj(feats), mask))


class AdditiveDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        content_vocab: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        n_mels: int,
    ) -> None:
        super().__init__()
        self.z_proj = nn.Linear(latent_dim, hidden)
        self.content = nn.Embedding(content_vocab, hidden)
        nn.init.normal_(self.content.weight, std=0.02)
        self.pos = SinusoidalPositionEncoding(hidden)
        self.transformer = TransformerStack(hidden, layers, heads, dropout)
        self.out = nn.Linear(hidden, n_mels)

    def forward(
        self, z: torch.Tensor, content_ids: Optional[torch.Tensor], mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.z_proj(z)
        if content_ids is not None:
            h = h + self.content(content_ids)
        h = self.transformer(self.pos(h), mask)
        return self.out(h), h


class AdditiveVAE(nn.Module):
    def __init__(
        self,
        cfg: Dict,
        num_speakers: int,
        num_emotions: int,
        latent_dim: int,
        n_mels: int,
        content_vocab: int,
        speaker_margin: float = 0.2,
        speaker_scale: float = 30.0,
    ) -> None:
        super().__init__()
        frontend_dim = int(get(cfg, "model.frontend.output_dim", 768))
        emo_cfg = get(cfg, "model.emotion_encoder")
        spk_cfg = get(cfg, "model.speaker_encoder")
        dec_cfg = get(cfg, "model.decoder")

        self.latent_dim = latent_dim
        self.emotion_encoder = EmotionEncoder(
            frontend_dim,
            int(emo_cfg["hidden"]),
            int(emo_cfg["layers"]),
            int(emo_cfg["heads"]),
            float(emo_cfg["dropout"]),
            latent_dim,
        )
        self.speaker_encoder = SpeakerEncoder(
            frontend_dim,
            int(spk_cfg["hidden"]),
            latent_dim,
            float(spk_cfg["dropout"]),
        )
        self.decoder = AdditiveDecoder(
            latent_dim,
            content_vocab,
            int(dec_cfg["hidden"]),
            int(dec_cfg["layers"]),
            int(dec_cfg["heads"]),
            float(dec_cfg["dropout"]),
            n_mels,
        )

        # Training-only heads.
        self.speaker_head = AAMSoftmax(
            latent_dim, num_speakers, margin=speaker_margin, scale=speaker_scale
        )
        self.emotion_head = nn.Linear(latent_dim, num_emotions)
        self.adv_head = nn.Linear(latent_dim, num_speakers)
        self.probe_speaker = nn.Linear(int(dec_cfg["hidden"]), num_speakers)
        self.probe_emotion = nn.Linear(int(dec_cfg["hidden"]), num_emotions)

    def encode(
        self, feats: torch.Tensor, mask: torch.Tensor, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v_spk = self.speaker_encoder(feats, mask)
        mu, logvar = self.emotion_encoder(feats, mask)
        if sample and self.training:
            std = torch.exp(0.5 * logvar)
            v_emo = mu + std * torch.randn_like(std)
        else:
            v_emo = mu
        return v_spk, v_emo, mu, logvar

    def forward(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor,
        content_ids: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        v_spk, v_emo, mu, logvar = self.encode(feats, mask, sample=sample)
        z = v_spk.unsqueeze(1) + v_emo
        mel_hat, hidden = self.decoder(z, content_ids, mask)
        return {
            "mel_hat": mel_hat,
            "hidden": hidden,
            "v_spk": v_spk,
            "v_emo": v_emo,
            "v_emo_mu": mu,
            "v_emo_logvar": logvar,
            "v_emo_utt": masked_mean(v_emo, mask),
            "z": z,
        }

    def decode_from(self, v_spk: torch.Tensor, v_emo: torch.Tensor, content_ids, mask):
        return self.decoder(v_spk.unsqueeze(1) + v_emo, content_ids, mask)
