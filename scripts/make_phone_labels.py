"""Convert MFA TextGrid alignments into frame-level phone id arrays.

Output: data/phone_labels/<utt_id>.npy, one int64 id per 20 ms frame, aligned to
the same hop used for log-mel (default 320 samples at 16 kHz).

Usage:
    python scripts/make_phone_labels.py --config configs/stage1_ppg.yaml
"""

from __future__ import annotations

import argparse
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emod.config import get, load_config, load_phone_set, resolve_path  # noqa: E402
from emod.data.dataset import manifest_path  # noqa: E402
from emod.data.manifest import read_manifest  # noqa: E402

SILENCE = {"", "sil", "sp", "spn", "<eps>", "<sil>", "silence"}


def parse_textgrid(path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """Minimal long-format TextGrid parser; returns {tier_name: intervals}."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    tiers: Dict[str, List[Tuple[float, float, str]]] = {}
    name = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("name ="):
            name = line.split("=", 1)[1].strip().strip('"')
            tiers.setdefault(name, [])
        elif line.startswith("intervals [") or line.startswith("points ["):
            if name is None:
                i += 1
                continue
            xmin = xmax = 0.0
            text = ""
            for j in range(1, 4):
                if i + j < len(lines) and "=" in lines[i + j]:
                    key, value = lines[i + j].split("=", 1)
                    key, value = key.strip(), value.strip().strip('"')
                    if key == "xmin":
                        xmin = float(value)
                    elif key == "xmax":
                        xmax = float(value)
                    elif key == "text":
                        text = value
            tiers[name].append((xmin, xmax, text))
            i += 3
        i += 1
    return tiers


def choose_tier(tiers: Dict[str, List[Tuple[float, float, str]]]) -> List[Tuple[float, float, str]]:
    for candidate in ("phones", "phone", "Phonetic", "phonemes"):
        for key in tiers:
            if key.lower() == candidate.lower() and tiers[key]:
                return tiers[key]
    for value in tiers.values():
        if value:
            return value
    return []


def frame_labels(
    intervals: List[Tuple[float, float, str]],
    num_frames: int,
    sample_rate: int,
    hop_length: int,
    phone_to_id: Dict[str, int],
) -> np.ndarray:
    ends = [end for _, end, _ in intervals]
    labels = np.zeros(num_frames, dtype=np.int64)
    for frame in range(num_frames):
        t = (frame * hop_length + hop_length / 2.0) / sample_rate
        idx = min(bisect_right(ends, t), len(intervals) - 1)
        text = intervals[idx][2].strip()
        key = "SIL" if text.lower() in SILENCE else text
        labels[frame] = phone_to_id.get(key, phone_to_id.get("SIL", 0))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="MFA TextGrid -> frame phone ids")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0, help="debug: only N files")
    args = parser.parse_args()

    cfg = load_config(args.config, extras=[args.gpu] if args.gpu else None, overrides=args.set)
    phones = load_phone_set(get(cfg, "data.phone_set"))
    phone_to_id = {phone: i for i, phone in enumerate(phones)}

    sample_rate = int(get(cfg, "data.sample_rate", 16000))
    hop = int(get(cfg, "data.hop_length", 320))
    raw_root = resolve_path(get(cfg, "data.raw_root"))
    grid_root = resolve_path(get(cfg, "data.textgrid_root"))
    out_dir = resolve_path(get(cfg, "paths.phone_label_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("train", "val", "test"):
        path = manifest_path(cfg, split)
        if path.exists():
            rows.extend(read_manifest(path))

    done, skipped = 0, 0
    unknown = set()
    for row in rows:
        if args.limit and done >= args.limit:
            break
        wav_path = raw_root / row["wav_path"]
        grid_path = grid_root / Path(row["wav_path"]).with_suffix(".TextGrid")
        if not grid_path.exists():
            skipped += 1
            continue

        import soundfile as sf

        info = sf.info(str(wav_path))
        num_frames = info.frames // hop + 1
        intervals = choose_tier(parse_textgrid(grid_path))
        if not intervals:
            skipped += 1
            continue
        for _, _, text in intervals:
            if text and text.lower() not in SILENCE and text not in phone_to_id:
                unknown.add(text)
        labels = frame_labels(intervals, num_frames, sample_rate, hop, phone_to_id)
        np.save(out_dir / f"{row['utt_id']}.npy", labels)
        done += 1

    print(f"wrote {done} label files to {out_dir} (skipped {skipped})")
    if unknown:
        print(f"warning: {len(unknown)} unknown phones mapped to SIL: {sorted(unknown)[:10]}")


if __name__ == "__main__":
    main()
