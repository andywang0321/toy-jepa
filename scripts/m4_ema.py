"""M4: JEPA with an EMA target encoder.

The target is a stopped-gradient copy of the online encoder, updated outside the
optimizer. The run writes to ``runs/m4_ema/`` and uses the same post-hoc readout
protocol as M1.

Usage:
    uv run python scripts/m4_ema.py
"""

from __future__ import annotations

from pathlib import Path

from toy_jepa.jepa_experiment import JEPAConfig, run_jepa

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m4_ema"


def main() -> None:
    """Train EMA JEPA and the common readout."""
    run_jepa(JEPAConfig(variant="ema", run_dir=RUN_DIR))


if __name__ == "__main__":
    main()
