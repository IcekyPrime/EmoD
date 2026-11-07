"""Post-training direction calibration (stage 2 -> stage 3 / inference)."""

from __future__ import annotations

from typing import Callable, Dict

import torch

from emod.models.blocks import masked_mean
from emod.utils.geometry import l2_normalize


@torch.no_grad()
def estimate_neutral_vectors(
    encode_fn: Callable, loader, neutral_id: int, device
) -> Dict[int, torch.Tensor]:
    """Average neutral emotion vector per speaker, over the whole split."""
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        v_emo, mask, speaker_id, emotion_id = encode_fn(batch)
        selected = emotion_id == neutral_id
        for speaker in torch.unique(speaker_id[selected]).tolist():
            rows = selected & (speaker_id == speaker)
            pooled = masked_mean(v_emo[rows], mask[rows]).mean(dim=0)
            sums[speaker] = sums.get(speaker, torch.zeros_like(pooled)) + pooled
            counts[speaker] = counts.get(speaker, 0) + 1
    return {k: sums[k] / max(1, counts[k]) for k in sums}


@torch.no_grad()
def estimate_directions(
    encode_fn: Callable,
    loader,
    neutral_vectors: Dict[int, torch.Tensor],
    num_emotions: int,
    neutral_id: int,
    device,
) -> Dict[str, torch.Tensor]:
    """Return unit directions, average scales and per-emotion counts."""
    dim = next(iter(neutral_vectors.values())).shape[-1] if neutral_vectors else 0
    if dim == 0:
        raise RuntimeError("no neutral reference found; cannot calibrate directions")

    sum_vec = torch.zeros(num_emotions, dim)
    sum_norm = torch.zeros(num_emotions)
    counts = torch.zeros(num_emotions)
    fallback = torch.stack(list(neutral_vectors.values())).mean(dim=0)

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        v_emo, mask, speaker_id, emotion_id = encode_fn(batch)
        v_neu = torch.stack(
            [neutral_vectors.get(int(s), fallback) for s in speaker_id.tolist()]
        ).to(v_emo.device)
        delta = v_emo - v_neu.unsqueeze(1)
        pooled = masked_mean(delta, mask).detach().cpu()

        for emotion in torch.unique(emotion_id).tolist():
            if emotion == neutral_id:
                continue
            rows = (emotion_id == emotion).cpu()
            sum_vec[emotion] += pooled[rows].sum(dim=0)
            sum_norm[emotion] += pooled[rows].norm(dim=-1).sum()
            counts[emotion] += float(rows.sum())

    table = torch.zeros(num_emotions, dim)
    scale = torch.zeros(num_emotions)
    for emotion in range(num_emotions):
        if counts[emotion] > 0:
            table[emotion] = l2_normalize(sum_vec[emotion] / counts[emotion], dim=-1)
            scale[emotion] = sum_norm[emotion] / counts[emotion]
    return {"table": table, "scale": scale, "counts": counts}
