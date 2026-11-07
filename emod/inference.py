"""Function-level inference API.

The encoder returns vectors, not audio. A TTS backbone only needs to add
`V_x + V_emo` at its own frame rate, which keeps EmoD independent of any
particular vocoder or language model.

Example:
    enc = EmoDEncoder.from_checkpoint("checkpoints/stage2/final.pt")
    ref = enc.encode_reference("demo.wav")
    target = enc.target_from_label("happy", num_frames=ref.emotion_frames.shape[0], intensity=1.2)
    joint = enc.compose(ref, target)          # [T, d]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from emod.models.blocks import masked_mean
from emod.models.factory import build_stage2_inference_bundle, build_t2e
from emod.models.t2e import T2EModel
from emod.utils.audio import lengths_to_mask, load_audio
from emod.utils.checkpoint import load_checkpoint


@dataclass
class ReferenceEncoding:
    """What EmoD(S2E) extracts from a reference utterance."""

    speaker: torch.Tensor          # [d] timbre vector V_x
    emotion_frames: torch.Tensor   # [T, d] frame-level V_emo,x
    emotion_utterance: torch.Tensor  # [d] mean over frames
    frame_rate: float              # frames per second (default 50 Hz)
    duration: float = 0.0          # seconds


@dataclass
class TargetEmotion:
    """A speaker-independent target emotion trajectory."""

    name: str
    direction: torch.Tensor        # [d] unit direction
    scale: float                   # average intensity of this emotion
    frames: torch.Tensor           # [T, d] V_emo(t)
    frame_rate: float = 50.0


class EmoDEncoder:
    """Stage-2 encoder: audio -> (V_x, V_emo)."""

    def __init__(self, frontend, vae, artifacts: Dict, device: torch.device) -> None:
        self.frontend = frontend
        self.vae = vae
        self.artifacts = artifacts
        self.emotions: List[str] = list(artifacts["emotions"])
        self.sample_rate = int(artifacts.get("sample_rate", 16000))
        self.frame_rate = self.sample_rate / float(artifacts.get("hop_length", 320))
        self.device = device
        directions = artifacts.get("directions", {})
        self.directions = directions.get("table")
        self.scales = directions.get("scale")

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "EmoDEncoder":
        payload = load_checkpoint(path, expected_stage=2)
        frontend, vae, _, artifacts = build_stage2_inference_bundle(payload, device=device)
        return cls(frontend, vae, artifacts, torch.device(device))

    # -- reference ---------------------------------------------------------
    def _coerce_wav(self, wav) -> torch.Tensor:
        if isinstance(wav, (str, Path)):
            wav = load_audio(wav, self.sample_rate)
        wav = torch.as_tensor(wav, dtype=torch.float32, device=self.device)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        return wav

    @torch.no_grad()
    def encode_reference(self, wav) -> ReferenceEncoding:
        wav = self._coerce_wav(wav)
        lengths = torch.tensor([wav.numel()], device=self.device, dtype=torch.long)
        feats, feat_lengths = self.frontend(wav.unsqueeze(0), lengths)
        mask = lengths_to_mask(feat_lengths, feats.shape[1])
        v_spk = self.vae.speaker_encoder(feats, mask)[0]
        v_emo, _ = self.vae.emotion_encoder(feats, mask)
        frames = int(mask.sum().item())
        return ReferenceEncoding(
            speaker=v_spk,
            emotion_frames=v_emo[0, :frames],
            emotion_utterance=masked_mean(v_emo, mask)[0],
            frame_rate=self.frame_rate,
            duration=wav.numel() / self.sample_rate,
        )

    # -- targets -----------------------------------------------------------
    def target_from_label(
        self, emotion: str, num_frames: int, intensity: float = 1.0
    ) -> TargetEmotion:
        if self.directions is None:
            raise RuntimeError("checkpoint has no calibrated directions")
        if emotion not in self.emotions:
            raise KeyError(f"unknown emotion '{emotion}'; available: {self.emotions}")
        index = self.emotions.index(emotion)
        direction = self.directions[index].to(self.device)
        scale = float(self.scales[index])
        frames = (direction * (scale * float(intensity))).unsqueeze(0).repeat(num_frames, 1)
        return TargetEmotion(emotion, direction, scale, frames, self.frame_rate)

    def target_from_frames(self, frames: torch.Tensor, name: str = "custom") -> TargetEmotion:
        frames = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
        norm = frames.norm(dim=-1).mean().clamp_min(1e-8)
        return TargetEmotion(name, frames.mean(0) / norm, float(norm), frames, self.frame_rate)

    # -- composition -------------------------------------------------------
    def compose(self, ref: ReferenceEncoding, target: TargetEmotion) -> torch.Tensor:
        """V_emo(tar),x = V_emo(tar) + V_x  (frame counts are aligned to the reference)."""
        frames = target.frames.to(self.device)
        if frames.shape[0] != ref.emotion_frames.shape[0]:
            frames = self.resample(frames, ref.emotion_frames.shape[0])
        return frames + ref.speaker.unsqueeze(0)

    def compose_from_audio(self, ref: ReferenceEncoding, target_audio) -> torch.Tensor:
        """EmoRES/EmoSteer-style transfer: keep ref timbre, take target emotion from audio."""
        target = self.encode_reference(target_audio)
        direction, scale = self._unit(target.emotion_utterance)
        return self.compose(
            ref,
            TargetEmotion(
                name="reference",
                direction=direction,
                scale=scale,
                frames=target.emotion_frames,
                frame_rate=self.frame_rate,
            ),
        )

    @staticmethod
    def _unit(vector: torch.Tensor):
        norm = vector.norm().clamp_min(1e-8)
        return vector / norm, float(norm)

    def resample(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        if frames.shape[0] == num_frames:
            return frames
        out = F.interpolate(
            frames.T.unsqueeze(0), size=num_frames, mode="linear", align_corners=False
        )
        return out.squeeze(0).T

    def to_rate(self, frames: torch.Tensor, target_rate: float) -> torch.Tensor:
        """Linear resampling from EmoD's 50 Hz to the consumer's frame rate."""
        if abs(target_rate - self.frame_rate) < 1e-3:
            return frames
        ratio = float(target_rate) / self.frame_rate
        return self.resample(frames, max(1, int(round(frames.shape[0] * ratio))))

    def save_vectors(self, path: str | Path, ref: ReferenceEncoding) -> None:
        torch.save(
            {
                "speaker": ref.speaker.cpu(),
                "emotion_frames": ref.emotion_frames.cpu(),
                "emotion_utterance": ref.emotion_utterance.cpu(),
                "frame_rate": ref.frame_rate,
            },
            path,
        )


class EmoDT2E:
    """Stage-3 branch: phoneme sequence + emotion label -> V_emo(t)."""

    def __init__(self, model: T2EModel, artifacts: Dict, device: torch.device) -> None:
        self.model = model
        self.artifacts = artifacts
        self.emotions: List[str] = list(artifacts["emotions"])
        self.device = device
        self.directions = artifacts["directions"]["table"].to(device)
        self.scales = artifacts["directions"]["scale"].to(device)
        self.frame_rate = float(artifacts.get("sample_rate", 16000)) / float(
            artifacts.get("hop_length", 320)
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "EmoDT2E":
        payload = load_checkpoint(path, expected_stage=3)
        artifacts = payload["artifacts"]
        cfg = payload["config"]
        model = build_t2e(
            cfg,
            num_phones=int(artifacts["num_phones"]),
            num_emotions=len(artifacts["emotions"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
        )
        state = payload["model"]
        weights = {
            k[len("t2e.") :]: v for k, v in state.items() if k.startswith("t2e.")
        } or state
        model.load_state_dict(weights, strict=True)
        model.to(device).eval()
        return cls(model, artifacts, torch.device(device))

    @torch.no_grad()
    def from_phones(
        self,
        phones: Sequence[int] | torch.Tensor,
        emotion: str,
        durations: Optional[Sequence[int]] = None,
        intensity: float = 1.0,
    ) -> TargetEmotion:
        if emotion not in self.emotions:
            raise KeyError(f"unknown emotion '{emotion}'; available: {self.emotions}")
        index = self.emotions.index(emotion)
        ids = torch.as_tensor(phones, dtype=torch.long, device=self.device).unsqueeze(0)
        lengths = torch.tensor([ids.shape[1]], device=self.device)
        dur = None
        if durations is not None:
            dur = torch.as_tensor(durations, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model(
            ids,
            lengths,
            torch.tensor([index], device=self.device),
            self.directions,
            self.scales,
            durations=dur,
        )
        frames = out["v_emo"][0, : int(out["frame_lengths"][0])] * float(intensity)
        return TargetEmotion(
            name=emotion,
            direction=self.directions[index],
            scale=float(self.scales[index]),
            frames=frames,
            frame_rate=self.frame_rate,
        )

    def from_text(self, *_args, **_kwargs):
        raise NotImplementedError(
            "G2P is intentionally out of scope. Convert text to phone ids with your own "
            "front-end and call `from_phones`."
        )
