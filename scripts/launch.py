"""Build and run the `accelerate launch` command from a YAML profile.

`runtime.num_processes: auto` resolves to min(visible GPUs, 8), which is what
makes the same config work from 1 to 8 cards.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emod.config import PROJECT_ROOT, get, load_config, num_processes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch EmoD training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, extras=[args.gpu] if args.gpu else None, overrides=args.set)
    world = int(args.num_processes or num_processes(cfg))
    mixed = str(get(cfg, "runtime.mixed_precision", "no"))

    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(world),
        "--mixed_precision",
        mixed,
    ]
    if world > 1:
        cmd.append("--multi_gpu")
    cmd += ["tools/train.py", "--config", args.config]
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.set:
        cmd += ["--set", *args.set]
    if args.dry_run:
        cmd.append("--dry-run")

    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
