"""Train the phase-2 pixel-prediction arm."""

from __future__ import annotations

import os
from pathlib import Path

from toy_jepa.phase2.config import Config
from toy_jepa.phase2.experiment import run

ROOT = Path(__file__).resolve().parents[1]
SEED = int(os.environ.get("P2_SEED", "0"))
CONFIG = Config(
    run_dir=ROOT / "runs" / "p2" / "pixel" / f"seed_{SEED}",
    checkpoint_dir=ROOT / "checkpoints" / "p2" / "pixel" / f"seed_{SEED}",
    seed=SEED,
)


if __name__ == "__main__":
    run("pixel", CONFIG)
