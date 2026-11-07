"""Small stateless helpers shared by trainers and scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int, deterministic: bool = False, tf32: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and tf32:
        # TF32 is safe for the fp32 parts of fp16/bf16 training.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


def resolve_batch_plan(cfg: Dict[str, Any], world_size: int, num_emotions: int) -> Dict[str, int]:
    """Derive per-GPU batch and gradient accumulation from the effective batch.

    Stage 2 uses the S x E x K sampler, so its per-GPU batch is structural; other
    stages read `train.per_gpu_batch_size`. Accumulation absorbs the remaining
    factor, which is what makes the same config work for 1-8 GPUs.
    """
    stage = int(cfg["stage"])
    if stage == 2:
        sampler = cfg["sampler"]
        per_gpu = int(sampler["num_speakers"]) * num_emotions * int(sampler["per_cell"])
    else:
        per_gpu = int(cfg["train"]["per_gpu_batch_size"])

    effective = int(cfg["train"]["effective_batch_size"])
    denom = max(1, per_gpu * max(1, world_size))
    accum = max(1, int(round(effective / denom)))
    return {
        "per_gpu_batch_size": per_gpu,
        "grad_accum_steps": accum,
        "effective_batch_size": per_gpu * max(1, world_size) * accum,
    }


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"
