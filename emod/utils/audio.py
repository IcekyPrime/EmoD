"""Audio IO and log-mel feature extraction (50 Hz at 16 kHz, hop = 320)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.functional as AF


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Load a mono waveform at the requested sample rate. Returns [N] float32."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = AF.resample(wav, sr, sample_rate)
    return wav


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """[B] lengths -> [B, T] bool mask (True = valid)."""
    if max_len is None:
        max_len = int(lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def mel_lengths(wav_lengths: torch.Tensor, hop_length: int) -> torch.Tensor:
    return torch.div(wav_lengths, hop_length, rounding_mode="floor") + 1


class LogMelExtractor(nn.Module):
    """Stateless log-mel front-end; parameters are buffers, so `.to(device)` works."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
    ) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.mel = torchaudio_mel(sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max)

    def forward(
        self, wav: torch.Tensor, wav_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, N] -> ([B, T, n_mels], lengths [B])."""
        mel = self.mel(wav)  # [B, n_mels, T]
        mel = torch.log(mel.clamp_min(1e-5)).transpose(1, 2)
        if wav_lengths is None:
            lengths = torch.full(
                (wav.shape[0],), mel.shape[1], dtype=torch.long, device=wav.device
            )
        else:
            lengths = torch.clamp(
                mel_lengths(wav_lengths, self.hop_length), max=mel.shape[1]
            )
        return mel, lengths


def torchaudio_mel(sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max):
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=2.0,
        center=True,
        norm="slaney",
        mel_scale="slaney",
    )
