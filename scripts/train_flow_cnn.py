"""Train the flow-matching convolutional pixel arm."""

from __future__ import annotations

import os
from pathlib import Path

from toy_jepa.config import Config
from toy_jepa.experiment import run

ROOT = Path(__file__).resolve().parents[1]
SEED = int(os.environ.get("P2_SEED", "0"))
CONFIG = Config(
    run_dir=ROOT / "runs" / "p2" / "flow_cnn" / f"seed_{SEED}",
    checkpoint_dir=ROOT / "checkpoints" / "p2" / "flow_cnn" / f"seed_{SEED}",
    seed=SEED,
)


if __name__ == "__main__":
    run("flow_cnn", CONFIG)
