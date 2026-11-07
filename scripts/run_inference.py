"""Minimal inference demo for the vector API (no TTS backbone required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emod.inference import EmoDEncoder, EmoDT2E  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="EmoD inference demo")
    parser.add_argument("--stage2", default="checkpoints/stage2/final.pt")
    parser.add_argument("--stage3", default=None, help="optional T2E checkpoint")
    parser.add_argument("--ref", required=True, help="reference wav (timbre + emotion)")
    parser.add_argument("--target", default=None, help="optional wav whose emotion is transferred")
    parser.add_argument("--emotion", default="happy")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="optional .pt path for the vectors")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    encoder = EmoDEncoder.from_checkpoint(args.stage2, device=args.device)
    ref = encoder.encode_reference(args.ref)
    print(f"reference: V_x={tuple(ref.speaker.shape)} "
          f"V_emo={tuple(ref.emotion_frames.shape)} @ {ref.frame_rate:.1f} Hz")

    if args.target:
        joint = encoder.compose_from_audio(ref, args.target)
        print(f"audio-to-audio transfer: joint={tuple(joint.shape)}")
    else:
        target = encoder.target_from_label(
            args.emotion, ref.emotion_frames.shape[0], intensity=args.intensity
        )
        joint = encoder.compose(ref, target)
        print(f"label target '{args.emotion}' (lambda={args.intensity}): joint={tuple(joint.shape)}")

    if args.stage3:
        t2e = EmoDT2E.from_checkpoint(args.stage3, device=args.device)
        phones = [1, 2, 3, 4, 5]
        out = t2e.from_phones(phones, args.emotion, durations=[8, 8, 8, 8, 8])
        print(f"T2E from phones: {tuple(out.frames.shape)}")

    if args.out:
        torch.save({"joint": joint.cpu(), "speaker": ref.speaker.cpu()}, args.out)
        print(f"saved vectors to {args.out}")


if __name__ == "__main__":
    main()
