"""Project constants.

Data lives outside the repository so datasets and checkpoints do not bloat git.
Edit these defaults directly; the scripts do not take command-line arguments.
"""

from pathlib import Path

DATA_DIR: Path = Path.home() / "data"
MNIST_DIR: Path = DATA_DIR / "mnist"

NUM_SAMPLES: int = 10_000
NUM_FRAMES: int = 20
CANVAS_SIZE: int = 64
DIGIT_SIZE: int = 28
SPEED_RANGE: tuple[float, float] = (1.0, 3.0)
POSITION_SCALE: float = float(CANVAS_SIZE - DIGIT_SIZE)
VELOCITY_SCALE: float = float(SPEED_RANGE[1])
TRAIN_FRACTION: float = 0.8
VAL_FRACTION: float = 0.1
SEED: int = 0
VISUALIZE_SAMPLES: int = 3

# Common post-hoc readout: 128 -> 256 -> 8, trained on frozen features.
READOUT_PAIRS: tuple[int, int, int] = (6_000, 1_000, 1_000)
READOUT_STEPS: int = 2_000
READOUT_BATCH_SIZE: int = 256
READOUT_LR: float = 1e-3
READOUT_WEIGHT_DECAY: float = 1e-4
READOUT_EVAL_EVERY: int = 50
READOUT_PATIENCE: int = 500
READOUT_CONFIG: dict[str, int | float] = {
    "num_steps": READOUT_STEPS,
    "batch_size": READOUT_BATCH_SIZE,
    "learning_rate": READOUT_LR,
    "weight_decay": READOUT_WEIGHT_DECAY,
    "eval_every": READOUT_EVAL_EVERY,
    "patience": READOUT_PATIENCE,
}
