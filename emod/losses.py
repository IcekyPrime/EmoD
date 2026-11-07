"""Losses for the additive space.

All geometry losses assume:
    delta_t = V_emo,t - V_neu,x        (neutral reference of the same speaker)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from emod.utils.geometry import l2_normalize


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs().mean(dim=-1)
    weights = mask.to(diff.dtype)
    return (diff * weights).sum() / weights.sum().clamp_min(1.0)


def spectral_convergence(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred - target).norm(dim=-1)
    den = target.norm(dim=-1).clamp_min(1e-6)
    return (num / den).mean()


def reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    l1 = masked_l1(pred, target, mask)
    sc = spectral_convergence(pred * mask.unsqueeze(-1), target * mask.unsqueeze(-1))
    return {"rec": l1, "sc": sc, "total": l1 + 0.1 * sc}


def kl_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    per_dim = kl.mean(dim=(0, 1))
    return per_dim.clamp_min(float(free_bits)).mean()


def frame_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape)
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def neutral_reference(
    v_emo: torch.Tensor,
    mask: torch.Tensor,
    speaker_id: torch.Tensor,
    emotion_id: torch.Tensor,
    neutral_id: int,
) -> torch.Tensor:
    """Per-speaker neutral vector, broadcast to [B, T, d]."""
    out = torch.zeros_like(v_emo)
    for spk in torch.unique(speaker_id):
        rows = (speaker_id == spk) & (emotion_id == neutral_id)
        if not bool(rows.any()):
            continue
        sub_mask = mask[rows].to(v_emo.dtype)
        pooled = (v_emo[rows] * sub_mask.unsqueeze(-1)).sum(1) / sub_mask.sum(
            1, keepdim=True
        ).clamp_min(1.0)
        out[speaker_id == spk] = pooled.mean(0, keepdim=True)
    return out


def cross_speaker_delta_loss(
    delta: torch.Tensor,
    mask: torch.Tensor,
    speaker_id: torch.Tensor,
    emotion_id: torch.Tensor,
    text_ids,
    neutral_id: int,
    frames: int = 16,
) -> torch.Tensor:
    """Same text + same emotion + different speakers should share one direction."""
    groups: Dict[tuple, list] = {}
    for i, (text, emo) in enumerate(zip(text_ids, emotion_id.tolist())):
        if emo == neutral_id:
            continue
        groups.setdefault((text, emo), []).append(i)

    losses = []
    for indices in groups.values():
        if len({int(speaker_id[i]) for i in indices}) < 2:
            continue
        vectors = []
        for i in indices:
            length = int(mask[i].sum().item())
            if length < 2:
                continue
            seq = delta[i, :length].T.unsqueeze(0)
            seq = F.interpolate(seq, size=frames, mode="linear", align_corners=False)
            vectors.append(l2_normalize(seq.squeeze(0).T, dim=-1))
        for a in range(len(vectors)):
            for b in range(a + 1, len(vectors)):
                losses.append(1.0 - (vectors[a] * vectors[b]).sum(-1).mean())

    if not losses:
        return delta.new_zeros(())
    return torch.stack(losses).mean()


def direction_compactness(
    delta: torch.Tensor,
    mask: torch.Tensor,
    direction_table: torch.Tensor,
    emotion_id: torch.Tensor,
    neutral_id: int,
) -> torch.Tensor:
    weights = mask.to(delta.dtype).unsqueeze(-1)
    pooled = (delta * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    cos = F.cosine_similarity(pooled, direction_table[emotion_id], dim=-1, eps=1e-8)
    valid = (emotion_id != neutral_id).to(cos.dtype)
    return ((1.0 - cos) * valid).sum() / valid.sum().clamp_min(1.0)


def orthogonality_loss(v_spk: torch.Tensor, v_emo_utt: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(v_spk, v_emo_utt, dim=-1, eps=1e-8).abs().mean()


def swap_probe_loss(
    model,
    v_spk: torch.Tensor,
    v_emo: torch.Tensor,
    content_ids: torch.Tensor,
    mask: torch.Tensor,
    speaker_id: torch.Tensor,
    emotion_id: torch.Tensor,
) -> torch.Tensor:
    """Decode swapped (V_x of A, V_emo of B) and require the probes to follow."""
    perm = torch.randperm(v_spk.shape[0], device=v_spk.device)
    _, hidden = model.decode_from(v_spk[perm], v_emo, content_ids, mask)
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    loss_spk = F.cross_entropy(model.probe_speaker(pooled), speaker_id[perm])
    loss_emo = F.cross_entropy(model.probe_emotion(pooled), emotion_id)
    return 0.5 * (loss_spk + loss_emo)


def scheduled_weight(
    base: float,
    step: int,
    start: int,
    ramp: int,
) -> float:
    """0 before `start`, linear ramp to `base` over `ramp` steps."""
    if step < start:
        return 0.0
    if ramp <= 0:
        return float(base)
    return float(base) * min(1.0, (step - start) / float(ramp))
