#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Physical GPU ids. Override with PIXEL_GPU=... JEPA_GPU=... if needed.
PIXEL_GPU="${PIXEL_GPU:-0}"
JEPA_GPU="${JEPA_GPU:-3}"

mkdir -p runs/p2_pixel runs/p2_jepa

pixel_pid=""
jepa_pid=""
cleanup() {
    for pid in "$pixel_pid" "$jepa_pid"; do
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM

echo "pixel -> GPU $PIXEL_GPU"
CUDA_VISIBLE_DEVICES="$PIXEL_GPU" uv run python scripts/p2_pixel.py \
    >runs/p2_pixel/train.log 2>&1 &
pixel_pid=$!

echo "jepa  -> GPU $JEPA_GPU"
CUDA_VISIBLE_DEVICES="$JEPA_GPU" uv run python scripts/p2_jepa.py \
    >runs/p2_jepa/train.log 2>&1 &
jepa_pid=$!

pixel_status=0
jepa_status=0
wait "$pixel_pid" || pixel_status=$?
wait "$jepa_pid" || jepa_status=$?

if ((pixel_status != 0 || jepa_status != 0)); then
    echo "a phase-2 run failed; see runs/p2_pixel/train.log and runs/p2_jepa/train.log"
    exit 1
fi

uv run python scripts/p2_analyze.py
