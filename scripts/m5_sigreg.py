"""M5: EMA JEPA plus SIGReg on the context feature.

Adds ``weight * sigreg_loss(c_t)`` to the EMA JEPA objective. The SIGReg
projection key is fixed for the whole run. The run writes to
``runs/m5_sigreg/`` and uses the same post-hoc readout protocol as M1.

Usage:
    uv run python scripts/m5_sigreg.py
"""

from __future__ import annotations

from pathlib import Path

from toy_jepa.jepa_experiment import JEPAConfig, run_jepa

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m5_sigreg"


def main() -> None:
    """Train EMA + SIGReg JEPA and the common readout."""
    run_jepa(JEPAConfig(variant="sigreg", run_dir=RUN_DIR))


if __name__ == "__main__":
    main()
