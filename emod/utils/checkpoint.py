"""Unified checkpoint format shared by all three stages.

Every stage writes the same top-level keys, so stage N+1 can inspect or reuse
stage N's artifacts without stage-specific loading code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

CKPT_FORMAT_VERSION = 1


def build_payload(
    stage: int,
    model: torch.nn.Module,
    epoch: int,
    step: int,
    cfg: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "format_version": CKPT_FORMAT_VERSION,
        "stage": int(stage),
        "epoch": int(epoch),
        "step": int(step),
        "config": cfg,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "artifacts": artifacts or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return payload


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    expected_stage: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != CKPT_FORMAT_VERSION:
        raise RuntimeError(
            f"unsupported checkpoint format {version} (expected {CKPT_FORMAT_VERSION})"
        )
    if expected_stage is not None and int(payload.get("stage", -1)) != int(expected_stage):
        raise RuntimeError(
            f"checkpoint stage mismatch: got {payload.get('stage')}, expected {expected_stage}"
        )
    return payload


def unwrap_state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = payload["model"]
    # Tolerate checkpoints saved through DDP/accelerate wrappers.
    return {k.replace("module.", "", 1): v for k, v in state.items()}


def latest_checkpoint(directory: str | Path) -> Optional[Path]:
    directory = Path(directory)
    last = directory / "last.pt"
    return last if last.exists() else None
