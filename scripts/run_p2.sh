#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/p2_pixel.py
uv run python scripts/p2_jepa.py
uv run python scripts/p2_analyze.py
