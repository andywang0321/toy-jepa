#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIXEL_GPU="${PIXEL_GPU:-0}"
JEPA_GPU="${JEPA_GPU:-3}"
FLOW_GPU="${FLOW_GPU:-2}"
SEEDS="${SEEDS:-0 1 2 3 4}"

pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
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

if [ "$PIXEL_GPU" = "$JEPA_GPU" ] && [ "$JEPA_GPU" = "$FLOW_GPU" ]; then
    # One GPU: run everything sequentially.
    run_arm pixel "$PIXEL_GPU" train_pixel.py
    run_arm jepa "$JEPA_GPU" train_jepa.py
    run_arm pixel_fm "$FLOW_GPU" train_pixel_flow.py
elif [ "$PIXEL_GPU" != "$JEPA_GPU" ] && [ "$PIXEL_GPU" != "$FLOW_GPU" ] && [ "$JEPA_GPU" != "$FLOW_GPU" ]; then
    # Three GPUs: one arm per GPU.
    run_arm pixel "$PIXEL_GPU" train_pixel.py &
    pids+=("$!")
    run_arm jepa "$JEPA_GPU" train_jepa.py &
    pids+=("$!")
    run_arm pixel_fm "$FLOW_GPU" train_pixel_flow.py &
    pids+=("$!")
else
    echo "Set PIXEL_GPU/JEPA_GPU/FLOW_GPU all equal (serial) or all distinct (parallel)." >&2
    exit 2
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if ((status != 0)); then
    echo "a phase-2 seed sweep failed"
    exit 1
fi

uv run python scripts/analyze.py
