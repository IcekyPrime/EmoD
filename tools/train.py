"""Unified training entry point.

Examples:
    accelerate launch --num_processes 4 tools/train.py --config configs/stage2_vae.yaml
    python tools/train.py --config configs/stage2_vae.yaml --gpu configs/gpu/h200_8gpu.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emod.config import load_config  # noqa: E402
from emod.trainers import build_trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train one EmoD stage")
    parser.add_argument("--config", required=True, help="stage config (yaml)")
    parser.add_argument("--gpu", default=None, help="optional GPU profile to merge on top")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=1e-5")
    parser.add_argument("--dry-run", action="store_true", help="two steps, no long run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.config,
        extras=[args.gpu] if args.gpu else None,
        overrides=args.set,
    )
    trainer = build_trainer(cfg, dry_run=args.dry_run)
    trainer.run()


if __name__ == "__main__":
    main()
