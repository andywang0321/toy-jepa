"""M0: overfit a small MLP to eight Moving MNIST frames.

This is a plumbing check, not a benchmark. It exercises the data pipeline,
MLP initialization, AdamW, and a jitted training step.

Usage:

    uv run python scripts/m0_overfit.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_jepa.constants import CANVAS_SIZE, DIGIT_SIZE, SEED, SPEED_RANGE
from toy_jepa.data.dataset import generate_dataset
from toy_jepa.models.mlp import init_mlp, mlp_mse_loss
from toy_jepa.core.optim import adamw_init
from toy_jepa.core.train import train_step
from toy_jepa.core.types import MLPParams, MovingMNIST, TrainState

NUM_SAMPLES = 8
NUM_STEPS = 1_000
LOG_EVERY = 100
HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m0_overfit"


def _flatten_first_frame(dataset: MovingMNIST, num_samples: int) -> jax.Array:
    """Flatten the first frame of each sequence and scale to [0, 1]."""
    frames = dataset.frames[:num_samples, 0].reshape(num_samples, -1)
    return jnp.asarray(frames, dtype=jnp.float32) / 255.0


def _first_frame_state(dataset: MovingMNIST, num_samples: int) -> jax.Array:
    """Return normalized (x, y, vx, vy) for both digits at frame zero."""
    position = dataset.position[:num_samples, 0].reshape(num_samples, -1)
    velocity = dataset.velocity[:num_samples, 0].reshape(num_samples, -1)
    position = jnp.asarray(position, dtype=jnp.float32) / (CANVAS_SIZE - DIGIT_SIZE)
    velocity = jnp.asarray(velocity, dtype=jnp.float32) / SPEED_RANGE[1]
    return jnp.concatenate([position, velocity], axis=-1)


def make_batch(dataset: MovingMNIST, num_samples: int) -> tuple[jax.Array, jax.Array]:
    """Build the M0 regression batch: flattened frame -> normalized state."""
    return (
        _flatten_first_frame(dataset, num_samples),
        _first_frame_state(dataset, num_samples),
    )


def init_state(
    key: jax.Array, input_dim: int, output_dim: int
) -> TrainState[MLPParams]:
    """Initialize the MLP and its AdamW moments."""
    params: MLPParams = init_mlp(key, [input_dim, HIDDEN_DIM, output_dim])
    return TrainState(params=params, opt_state=adamw_init(params))


def train(
    x: jax.Array,
    y: jax.Array,
    *,
    key: jax.Array,
    steps: int,
    log_every: int,
    lr: float,
    weight_decay: float,
) -> tuple[TrainState[MLPParams], list[tuple[int, float]]]:
    """Run full-batch training and return the final state and loss history."""
    state = init_state(key, x.shape[1], y.shape[1])
    step = jax.jit(
        lambda state, batch: train_step(
            state, batch, mlp_mse_loss, lr=lr, weight_decay=weight_decay
        )
    )

    history: list[tuple[int, float]] = []
    for i in range(steps):
        state, loss = step(state, (x, y))
        if i % log_every == 0 or i == steps - 1:
            loss_value = float(loss)
            history.append((i, loss_value))
            print(f"step {i:4d}  loss {loss_value:.6e}")

    return state, history


def write_history(history: list[tuple[int, float]], path: Path) -> None:
    """Write the logged loss values to a small CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "loss"])
        writer.writerows(history)


def plot_loss(history: list[tuple[int, float]], path: Path) -> None:
    """Plot the loss curve on a log scale."""
    steps, losses = zip(*history)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(steps, losses, marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate the overfit batch, train, and write the logs."""
    dataset = generate_dataset(SEED, NUM_SAMPLES)
    x, y = make_batch(dataset, NUM_SAMPLES)

    print(f"frames: {dataset.frames.shape}; x: {x.shape}; y: {y.shape}")
    print(f"steps: {NUM_STEPS}; lr: {LEARNING_RATE}; hidden: {HIDDEN_DIM}")

    state, history = train(
        x,
        y,
        key=jax.random.key(SEED),
        steps=NUM_STEPS,
        log_every=LOG_EVERY,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    write_history(history, RUN_DIR / "metrics.csv")
    plot_loss(history, RUN_DIR / "loss.png")

    print(f"final loss: {history[-1][1]:.6e}")
    print(f"optimizer steps: {int(state.opt_state.t)}")
    print(f"wrote {RUN_DIR / 'metrics.csv'} and {RUN_DIR / 'loss.png'}")


if __name__ == "__main__":
    main()
