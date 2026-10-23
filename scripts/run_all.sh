#!/usr/bin/env bash
set -euo pipefail

echo "Experiments running on GPUs ${CUDA_VISIBLE_DEVICES:-}"

uv sync
uv run python scripts/m0_overfit.py
uv run python scripts/m1_pixel.py
uv run python scripts/m2_jepa.py
uv run python scripts/m3_stopgrad.py
uv run python scripts/m4_ema.py
uv run python scripts/m5_sigreg.py
uv run python scripts/sweep_ema.py
uv run python scripts/sweep_sigreg.py

echo "Experiments completed"
