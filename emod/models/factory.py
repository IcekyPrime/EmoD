"""Model builders and checkpoint-based loaders."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from emod.config import get
from emod.models.additive_vae import AdditiveVAE
from emod.models.dppg import DPPGTokenizer
from emod.models.frontend import build_frontend
from emod.models.ppg import build_ppg_head
from emod.models.t2e import build_t2e


def build_additive_vae(
    cfg: Dict,
    num_speakers: int,
    num_emotions: int,
    num_phones: int,
    content_vocab: Optional[int] = None,
) -> AdditiveVAE:
    latent_dim = int(get(cfg, "model.latent_dim"))
    n_mels = int(get(cfg, "data.n_mels"))
    if content_vocab is None:
        content_vocab = num_phones * int(get(cfg, "model.dppg.max_clusters"))
    return AdditiveVAE(
        cfg=cfg,
        num_speakers=num_speakers,
        num_emotions=num_emotions,
        latent_dim=latent_dim,
        n_mels=n_mels,
        content_vocab=content_vocab,
        speaker_margin=float(get(cfg, "model.speaker_loss.margin", 0.2)),
        speaker_scale=float(get(cfg, "model.speaker_loss.scale", 30.0)),
    )


def _load_payload(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        return checkpoint
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def _sub_state(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    key = prefix + "."
    return {k[len(key) :]: v for k, v in state.items() if k.startswith(key)}


def build_stage2_bundle(
    cfg: Optional[Dict],
    checkpoint: Any,
    device: str | torch.device = "cpu",
) -> Tuple[AdditiveVAE, DPPGTokenizer, Dict[str, Any]]:
    """Rebuild the additive VAE from a stage-2 checkpoint (weights under `vae.*`)."""
    payload = _load_payload(checkpoint)
    artifacts = payload["artifacts"]
    source_cfg = payload.get("config", {})
    cfg = cfg or {}

    merged = dict(source_cfg)
    merged["model"] = {**source_cfg.get("model", {}), **cfg.get("model", {})}
    merged["data"] = {**source_cfg.get("data", {}), **cfg.get("data", {})}

    model = build_additive_vae(
        merged,
        num_speakers=len(artifacts["speakers"]),
        num_emotions=len(artifacts["emotions"]),
        num_phones=int(artifacts["num_phones"]),
        content_vocab=int(artifacts["content_vocab"]),
    )
    state = payload["model"]
    model.load_state_dict(
        _sub_state(state, "vae") if any(k.startswith("vae.") for k in state) else state,
        strict=True,
    )
    model.to(device)

    tokenizer = DPPGTokenizer.from_state(artifacts["dppg"])
    return model, tokenizer, artifacts


def build_stage2_inference_bundle(
    checkpoint: Any, device: str | torch.device = "cpu"
) -> Tuple[torch.nn.Module, AdditiveVAE, DPPGTokenizer, Dict[str, Any]]:
    """Front-end + additive VAE + tokenizer, ready for `EmoDEncoder`."""
    payload = _load_payload(checkpoint)
    artifacts = payload["artifacts"]
    cfg = payload["config"]

    frontend = build_frontend(cfg)
    state = payload["model"]
    frontend.load_state_dict(_sub_state(state, "frontend"), strict=True)

    vae = build_additive_vae(
        cfg,
        num_speakers=len(artifacts["speakers"]),
        num_emotions=len(artifacts["emotions"]),
        num_phones=int(artifacts["num_phones"]),
        content_vocab=int(artifacts["content_vocab"]),
    )
    vae.load_state_dict(_sub_state(state, "vae"), strict=True)

    frontend.to(device).eval()
    vae.to(device).eval()
    tokenizer = DPPGTokenizer.from_state(artifacts["dppg"])
    return frontend, vae, tokenizer, artifacts


def build_stage3_model(cfg: Dict, artifacts: Dict[str, Any], device="cpu"):
    return build_t2e(
        cfg,
        num_phones=int(artifacts["num_phones"]),
        num_emotions=len(artifacts["emotions"]),
        latent_dim=int(get(cfg, "model.latent_dim")),
    ).to(device)


__all__ = [
    "build_frontend",
    "build_ppg_head",
    "build_t2e",
    "build_additive_vae",
    "build_stage2_bundle",
    "build_stage2_inference_bundle",
    "build_stage3_model",
]
