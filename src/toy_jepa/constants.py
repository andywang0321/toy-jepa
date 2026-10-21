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
TRAIN_FRACTION: float = 0.8
VAL_FRACTION: float = 0.1
SEED: int = 0
VISUALIZE_SAMPLES: int = 3
