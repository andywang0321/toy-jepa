"""Small, explicit phase-2 experiment settings."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

M_TRAIN = (2, 4, 8)
M_TEST = (3, 6, 12)
K_TRAIN = (1, 2, 4, 8, 16, 32)
K_TEST = (3, 6, 12, 24, 48)
HORIZON_SCALE = 64.0
FINAL_PROB = 0.25

EMBED_DIM = 128
HEADS = 4
LAYERS = 2
MLP_RATIO = 4
MAX_M = max(M_TRAIN + M_TEST)
PIXEL_HIDDEN = 512


class Config(NamedTuple):
    run_dir: Path
    seed: int = 0
    num_samples: int = 2_000
    num_steps: int = 3_000
    batch_size: int = 32
    eval_every: int = 250
    eval_batches: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    readout_steps: int = 1_500
    readout_batch_size: int = 256
    readout_lr: float = 1e-3
    readout_pairs: int = 2_000
    eval_pairs_per_k: int = 256
    final_pairs: int = 512
