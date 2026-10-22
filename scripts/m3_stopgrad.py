"""M3: JEPA with a stop-gradient target.

Same model and data as M2, but gradients do not flow through the shared
encoder's target. The run writes to ``runs/m3_stopgrad/`` and uses the same
post-hoc readout protocol as M1.

Usage:
    uv run python scripts/m3_stopgrad.py
"""

from __future__ import annotations

from pathlib import Path

from toy_jepa.jepa_experiment import JEPAConfig, run_jepa

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m3_stopgrad"


def main() -> None:
    """Train stop-grad JEPA and the common readout."""
    run_jepa(JEPAConfig(variant="stopgrad", run_dir=RUN_DIR))


if __name__ == "__main__":
    main()
