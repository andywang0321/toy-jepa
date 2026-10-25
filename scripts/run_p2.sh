#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIXEL_GPU="${PIXEL_GPU:-0}"
JEPA_GPU="${JEPA_GPU:-3}"
SEEDS="${SEEDS:-0 1}"

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

run_arm() {
    local name="$1"
    local gpu="$2"
    local script="$3"
    local seed
    for seed in $SEEDS; do
        mkdir -p "runs/p2/$name/seed_$seed"
        echo "$name seed $seed -> GPU $gpu"
        if ! P2_SEED="$seed" CUDA_VISIBLE_DEVICES="$gpu" \
            uv run python "scripts/$script" \
            >"runs/p2/$name/seed_$seed/train.log" 2>&1; then
            echo "$name seed $seed failed; see runs/p2/$name/seed_$seed/train.log"
            return 1
        fi
    done
}

run_arm pixel "$PIXEL_GPU" p2_pixel.py &
pixel_pid=$!
run_arm jepa "$JEPA_GPU" p2_jepa.py &
jepa_pid=$!

pixel_status=0
jepa_status=0
wait "$pixel_pid" || pixel_status=$?
wait "$jepa_pid" || jepa_status=$?

if ((pixel_status != 0 || jepa_status != 0)); then
    echo "a phase-2 seed sweep failed"
    exit 1
fi

uv run python scripts/p2_analyze.py
