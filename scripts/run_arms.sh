#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/m0_overfit.py
uv run python scripts/m1_pixel.py
uv run python scripts/m2_jepa.py
uv run python scripts/m3_stopgrad.py
uv run python scripts/m4_ema.py
uv run python scripts/m5_sigreg.py
uv run python scripts/analyze.py
