"""Run every M1-M5 arm at five seeds and 5,000 steps."""

from __future__ import annotations

from pathlib import Path

import m1_pixel
import m2_jepa
import m3_stopgrad
import m4_ema
import m5_sigreg

SEEDS = tuple(range(5))
NUM_STEPS = 5_000
OUT = Path(__file__).resolve().parents[1] / "runs" / "seeds"


def run_arm(arm: str, module, seed: int) -> None:
    """Run one arm with one seed."""
    config = module.Config(
        num_steps=NUM_STEPS,
        seed=seed,
        run_dir=OUT / arm / f"seed_{seed}",
    )
    print(f"=== {arm} seed {seed} ===")
    module.run(config)


def main() -> None:
    for seed in SEEDS:
        run_arm("m1_pixel", m1_pixel, seed)
        run_arm("m2_jepa", m2_jepa, seed)
        run_arm("m3_stopgrad", m3_stopgrad, seed)
        run_arm("m4_ema", m4_ema, seed)
        run_arm("m5_sigreg", m5_sigreg, seed)


if __name__ == "__main__":
    main()
