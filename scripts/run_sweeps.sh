#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/sweep_ema.py
uv run python scripts/sweep_sigreg.py
uv run python scripts/analyze.py

echo "Sweeps finished. Inspect runs/analysis/report.md."
echo "Set EMA_DECAY in scripts/m4_ema.py and SIGREG_WEIGHT in scripts/m5_sigreg.py,"
echo "then run scripts/run_arms.sh."
