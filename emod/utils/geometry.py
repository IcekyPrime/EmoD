"""Geometry of the additive latent space.

Conventions
-----------
V_x      : speaker (timbre) vector, [d]
V_emo_t  : frame-level emotion vector, [T, d]
delta    : V_emo_t - V_emo_neu, i.e. the speaker-independent emotion offset
v_hat    : unit emotion direction, [d]
s        : average intensity scale for that emotion, scalar
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Sequence


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=dim, eps=1e-8)


def project_intensity(
    delta: torch.Tensor, direction: torch.Tensor, scale: float | torch.Tensor
) -> torch.Tensor:
    """Intensity = projection of the emotion offset onto the unit direction.

    Args:
        delta: [..., d] emotion offset.
        direction: [d] unit direction.
        scale: per-emotion average intensity, used to make lambda = 1 the mean.
    Returns:
        [...] intensity coefficients.
    """
    scale = torch.as_tensor(scale, dtype=delta.dtype, device=delta.device).clamp_min(1e-8)
    return (delta * direction).sum(dim=-1) / scale


def weighted_direction(
    weights: Sequence[float] | Dict[str, float],
    table: torch.Tensor,
    names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Normalized weighted sum of emotion directions (mixture target)."""
    if isinstance(weights, dict):
        if names is None:
            raise ValueError("names are required when weights is a mapping")
        vec = torch.zeros_like(table[0])
        for name, w in weights.items():
            vec = vec + float(w) * table[names.index(name)]
    else:
        w = torch.as_tensor(weights, dtype=table.dtype, device=table.device)
        if w.numel() != table.shape[0]:
            raise ValueError("weights and direction table size mismatch")
        vec = (w.unsqueeze(-1) * table).sum(dim=0)
    return l2_normalize(vec, dim=0)


def additivity_error(
    delta: torch.Tensor, direction: torch.Tensor, scale: float
) -> torch.Tensor:
    """Relative residual ||delta - s * v_hat|| / ||delta||."""
    target = float(scale) * direction
    return (delta - target).norm(dim=-1) / delta.norm(dim=-1).clamp_min(1e-8)


def mean_direction(
    deltas: torch.Tensor, masks: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Average a set of offsets and return (unit direction, mean norm)."""
    if masks is not None:
        w = masks.to(deltas.dtype).unsqueeze(-1)
        mean = (deltas * w).sum(dim=0) / w.sum(dim=0).clamp_min(1.0)
        scale = (deltas.norm(dim=-1) * masks).sum() / masks.sum().clamp_min(1)
    else:
        mean = deltas.mean(dim=0)
        scale = deltas.norm(dim=-1).mean()
    return l2_normalize(mean, dim=-1), scale
