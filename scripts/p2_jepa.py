"""Train the phase-2 stop-gradient JEPA arm."""

from __future__ import annotations

from pathlib import Path

from toy_jepa.phase2.config import Config
from toy_jepa.phase2.experiment import run

CONFIG = Config(run_dir=Path(__file__).resolve().parents[1] / "runs" / "p2_jepa")


if __name__ == "__main__":
    run("jepa", CONFIG)
