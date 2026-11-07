"""Scan a raw corpus and write train/val/test manifests (dataset-agnostic)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emod.config import as_rel, load_config  # noqa: E402
from emod.data.base import build_manifests, available_datasets  # noqa: E402
from emod.data.manifest import read_manifest, manifest_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EmoD manifests")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, extras=[args.gpu] if args.gpu else None, overrides=args.set)
    written = build_manifests(cfg)
    print(f"dataset adapters: {available_datasets()}")
    for split, path in written.items():
        rows = read_manifest(path)
        stats = manifest_stats(rows)
        print(
            f"[{split}] {as_rel(path)} | utt={stats['utterances']} "
            f"spk={stats['speakers']} emo={stats['emotions']} hours={stats['hours']}"
        )


if __name__ == "__main__":
    main()
