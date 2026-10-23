#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/sweep_seeds.py
uv run python scripts/analyze_seeds.py
