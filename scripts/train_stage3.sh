#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/stage3_t2e.yaml}"
GPU="${2:-configs/gpu/rtx3090_4gpu.yaml}"
if [ "$#" -ge 2 ]; then
  shift 2
elif [ "$#" -eq 1 ]; then
  shift 1
fi

python scripts/launch.py --config "$CONFIG" --gpu "$GPU" "$@"
