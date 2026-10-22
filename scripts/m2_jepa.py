"""M2: barebones JEPA, no target asymmetry.

Shared encoder for context and target, no stop-gradient. The run writes to
``runs/m2_jepa/`` with the same post-hoc state-readout protocol as M1.

Usage:
    uv run python scripts/m2_jepa.py
"""

from __future__ import annotations

from pathlib import Path

from toy_jepa.jepa_experiment import JEPAConfig, run_jepa

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m2_jepa"


def main() -> None:
    """Train barebones JEPA and the common readout."""
    run_jepa(JEPAConfig(variant="base", run_dir=RUN_DIR))


if __name__ == "__main__":
    main()
