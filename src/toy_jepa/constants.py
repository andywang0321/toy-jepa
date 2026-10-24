"""Project constants. Edit defaults directly; scripts take no CLI arguments."""

NUM_SAMPLES: int = 10_000
CANVAS_SIZE: int = 64
DIGIT_SIZE: int = 28
SPEED_RANGE: tuple[float, float] = (1.0, 3.0)
POSITION_SCALE: float = float(CANVAS_SIZE - DIGIT_SIZE)
VELOCITY_SCALE: float = float(SPEED_RANGE[1])
SEED: int = 0
VISUALIZE_SAMPLES: int = 3

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
