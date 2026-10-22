"""M1: two-headed baseline on Moving MNIST.

Context is two frames, ``(x_{t-1}, x_t)``. The model has two heads:

- a pixel head that predicts ``x_{t+k}``;
- a state head that predicts the target-time state of both digits:
  ``(x, y, vx, vy)`` for each digit.

The joint loss is pixel MSE plus a weighted state MSE. The state head is
Baseline B: an upper bound on how much state information the shared encoder
can carry.

Usage:
    uv run python scripts/m1_baseline.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toy_jepa.constants import CANVAS_SIZE, DIGIT_SIZE, SEED, SPEED_RANGE
from toy_jepa.data import (
    generate_dataset,
    make_context_target,
    pair_indices,
    state_at,
)
from toy_jepa.metrics import (
    effective_rank,
    feature_variance,
    knn_accuracy,
    mean_pairwise_cosine,
    ridge_r2,
)
from toy_jepa.models.baseline import apply_baseline, init_baseline
from toy_jepa.optim import adamw_init
from toy_jepa.train import baseline_loss, train_step
from toy_jepa.types import (
    BaselineParams,
    Float32Array,
    Int8Array,
    Int32Array,
    MovingMNIST,
    TrainState,
    UInt8Array,
)

NUM_SAMPLES = 2_000
K = 4
BATCH_SIZE = 64
NUM_STEPS = 1_500
EVAL_EVERY = 250
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
STATE_WEIGHT = 1.0
EMBED_DIM = 128
HIDDEN_DIM = 256
EVAL_PAIRS = 256
PROBE_PAIRS = 1_000
FEATURE_RIDGE_ALPHA = 1.0
RAW_RIDGE_ALPHA = 100.0
POSITION_SCALE = float(CANVAS_SIZE - DIGIT_SIZE)
VELOCITY_SCALE = float(SPEED_RANGE[1])
RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m1_baseline"


def _normalize_frames(frames: UInt8Array) -> jax.Array:
    """Scale uint8 frames to float32 in [0, 1]."""
    return jnp.asarray(frames, dtype=jnp.float32) / 255.0


def _normalize_state(
    position: Float32Array,
    velocity: Float32Array,
) -> jax.Array:
    """Scale both digits' (x, y, vx, vy) to comparable ranges."""
    position_array = jnp.asarray(position, dtype=jnp.float32) / POSITION_SCALE
    velocity_array = jnp.asarray(velocity, dtype=jnp.float32) / VELOCITY_SCALE
    return jnp.concatenate([position_array, velocity_array], axis=-1)


def _batch_from_indices(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    context, target_pixels = make_context_target(
        dataset, sequence_idx, time_idx, k
    )
    position, velocity, _ = state_at(dataset, sequence_idx, time_idx + k)
    return (
        _normalize_frames(context),
        _normalize_frames(target_pixels),
        _normalize_state(position, velocity),
    )


def sample_batch(
    rng: np.random.Generator,
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample context, target pixels, and target state for training."""
    choice = rng.integers(0, len(sequence_idx), size=batch_size)
    return _batch_from_indices(
        dataset, sequence_idx[choice], time_idx[choice], k
    )


def fixed_batch(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
    num_pairs: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build a deterministic evaluation batch."""
    return _batch_from_indices(
        dataset, sequence_idx[:num_pairs], time_idx[:num_pairs], k
    )


def probe_batch(
    dataset: MovingMNIST,
    rng: np.random.Generator,
    split: int,
    k: int,
    num_pairs: int,
) -> tuple[jax.Array, Float32Array, Float32Array, Int8Array]:
    """Sample context frames plus their current state labels for probes."""
    sequence_idx, time_idx = pair_indices(dataset, split, k)
    count = min(num_pairs, len(sequence_idx))
    choice = rng.choice(len(sequence_idx), size=count, replace=False)
    sequence_idx = sequence_idx[choice]
    time_idx = time_idx[choice]

    context, _ = make_context_target(dataset, sequence_idx, time_idx, k)
    position, velocity, labels = state_at(dataset, sequence_idx, time_idx)
    return (
        _normalize_frames(context),
        position,
        velocity,
        labels,
    )


def init_state(key: jax.Array) -> TrainState[BaselineParams]:
    """Initialize the baseline model and its AdamW moments."""
    params = init_baseline(
        key, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    )
    return TrainState(params=params, opt_state=adamw_init(params))


def predict(
    params: BaselineParams, context: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return predicted pixels, predicted state, and context features."""
    pixels, state, z_context, _ = apply_baseline(params, context)
    return pixels, state, z_context


predict_jit = jax.jit(predict)


def evaluate_baseline(
    params: BaselineParams,
    context: jax.Array,
    target_pixels: jax.Array,
    target_state: jax.Array,
) -> tuple[float, float, float, float, float]:
    """Return pixel MSE, state MSE, persistence MSE, and state MAEs."""
    pixels, state, _ = predict_jit(params, context)
    pixel_mse = jnp.mean((pixels - target_pixels) ** 2)
    state_mse = jnp.mean((state - target_state) ** 2)
    # Persistence: assume nothing changes, so x_{t+k} is just the current x_t.
    persistence = jnp.mean((context[:, 1] - target_pixels) ** 2)

    state_error = jnp.abs(state - target_state)
    position_mae = jnp.mean(state_error[:, :4]) * POSITION_SCALE
    velocity_mae = jnp.mean(state_error[:, 4:]) * VELOCITY_SCALE
    return (
        float(pixel_mse),
        float(state_mse),
        float(persistence),
        float(position_mae),
        float(velocity_mae),
    )


def train(
    dataset: MovingMNIST,
    train_sequence: Int32Array,
    train_time: Int32Array,
    val_context: jax.Array,
    val_pixels: jax.Array,
    val_state: jax.Array,
    key: jax.Array,
) -> tuple[TrainState[BaselineParams], list[tuple[int, float, float, float, float]]]:
    """Train both heads and log the total, pixel, and state losses."""
    state = init_state(key)
    rng = np.random.default_rng(SEED + 1)

    def loss_fn(
        params: BaselineParams,
        batch: tuple[jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        return baseline_loss(params, batch, state_weight=STATE_WEIGHT)

    @jax.jit
    def step(
        state: TrainState[BaselineParams],
        batch: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[TrainState[BaselineParams], jax.Array]:
        return train_step(
            state,
            batch,
            loss_fn,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    history: list[tuple[int, float, float, float, float]] = []
    for i in range(NUM_STEPS):
        context, target_pixels, target_state = sample_batch(
            rng, dataset, train_sequence, train_time, K, BATCH_SIZE
        )
        state, loss = step(state, (context, target_pixels, target_state))

        if i % EVAL_EVERY == 0 or i == NUM_STEPS - 1:
            val_pixel_mse, val_state_mse, persistence, _, _ = (
                evaluate_baseline(
                    state.params, val_context, val_pixels, val_state
                )
            )
            row = (i, float(loss), val_pixel_mse, val_state_mse, persistence)
            history.append(row)
            print(
                f"step {i:4d}  train {row[1]:.6f}  "
                f"val pixel {val_pixel_mse:.6f}  "
                f"val state {val_state_mse:.6f}  "
                f"persistence {persistence:.6f}"
            )

    return state, history


def compute_probes(
    state: TrainState[BaselineParams],
    dataset: MovingMNIST,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Fit probes on train contexts and evaluate on test contexts."""
    train_context, train_pos, train_vel, train_labels = probe_batch(
        dataset, rng, split=0, k=K, num_pairs=PROBE_PAIRS
    )
    test_context, test_pos, test_vel, test_labels = probe_batch(
        dataset, rng, split=2, k=K, num_pairs=PROBE_PAIRS
    )

    _, _, train_features = predict_jit(state.params, train_context)
    _, _, test_features = predict_jit(state.params, test_context)
    train_features = np.asarray(train_features)
    test_features = np.asarray(test_features)

    raw_train = np.asarray(train_context[:, 1]).reshape(len(train_context), -1)
    raw_test = np.asarray(test_context[:, 1]).reshape(len(test_context), -1)

    return {
        "position_r2": ridge_r2(
            train_features, train_pos, test_features, test_pos,
            alpha=FEATURE_RIDGE_ALPHA,
        ),
        "velocity_r2": ridge_r2(
            train_features, train_vel, test_features, test_vel,
            alpha=FEATURE_RIDGE_ALPHA,
        ),
        "digit_accuracy": knn_accuracy(
            train_features, train_labels, test_features, test_labels
        ),
        "raw_position_r2": ridge_r2(
            raw_train, train_pos, raw_test, test_pos,
            alpha=RAW_RIDGE_ALPHA,
        ),
        "raw_velocity_r2": ridge_r2(
            raw_train, train_vel, raw_test, test_vel,
            alpha=RAW_RIDGE_ALPHA,
        ),
        "raw_digit_accuracy": knn_accuracy(
            raw_train, train_labels, raw_test, test_labels
        ),
        "effective_rank": effective_rank(test_features),
        "feature_variance": feature_variance(test_features),
        "mean_pairwise_cosine": mean_pairwise_cosine(test_features),
        "probe_pairs": len(test_features),
    }


def write_history(
    history: list[tuple[int, float, float, float, float]], path: Path
) -> None:
    """Write train/val metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["step", "train_loss", "val_pixel_mse", "val_state_mse", "persistence_mse"]
        )
        writer.writerows(history)


def write_metrics(metrics: dict[str, float | int], path: Path) -> None:
    """Write final probe/evaluation metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def write_config(path: Path) -> None:
    """Write the run configuration for reproducibility."""
    config = {
        "num_samples": NUM_SAMPLES,
        "k": K,
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "eval_every": EVAL_EVERY,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "state_weight": STATE_WEIGHT,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "eval_pairs": EVAL_PAIRS,
        "probe_pairs": PROBE_PAIRS,
        "feature_ridge_alpha": FEATURE_RIDGE_ALPHA,
        "raw_ridge_alpha": RAW_RIDGE_ALPHA,
        "position_scale": POSITION_SCALE,
        "velocity_scale": VELOCITY_SCALE,
        "seed": SEED,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def plot_loss(
    history: list[tuple[int, float, float, float, float]], path: Path
) -> None:
    """Plot total/pixel losses and the normalized state loss."""
    steps, train_losses, val_pixels, val_states, persistence = zip(*history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axes[0].plot(steps, train_losses, marker="o", markersize=3, label="train total")
    axes[0].plot(steps, val_pixels, marker="o", markersize=3, label="val pixel")
    axes[0].plot(steps, persistence, linestyle="--", label="persistence")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, val_states, marker="o", markersize=3, color="tab:green")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("state MSE (normalized)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(
    params: BaselineParams,
    context: jax.Array,
    target_pixels: jax.Array,
    path: Path,
) -> None:
    """Plot current frame, target, prediction, and persistence."""
    count = min(4, len(context))
    pixels, _, _ = predict_jit(params, context[:count])
    pixels_np = np.asarray(pixels)
    context_np = np.asarray(context)
    target_np = np.asarray(target_pixels)

    rows = [
        ("current  $x_t$", context_np[:count, 1]),
        ("target  $x_{t+k}$", target_np[:count]),
        ("prediction", pixels_np),
        ("persistence  $x_t$", context_np[:count, 1]),
    ]
    fig, axes = plt.subplots(4, count, figsize=(3 * count, 10))
    fig.subplots_adjust(
        left=0.10, right=0.99, top=0.91, bottom=0.02, hspace=0.05, wspace=0.04
    )

    for row, (_, images) in enumerate(rows):
        for column in range(count):
            axis = axes[row, column]
            axis.imshow(images[column], cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row == 0:
                axis.set_title(f"sample {column}")

    # Draw row labels as figure text so they survive axis("off").
    row_height = (0.91 - 0.02) / len(rows)
    for row, (name, _) in enumerate(rows):
        y = 0.91 - (row + 0.5) * row_height
        fig.text(
            0.02,
            y,
            name,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )

    fig.suptitle("Baseline pixel predictions (columns are test samples)", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_probes(metrics: dict[str, float | int], path: Path) -> None:
    """Compare probe R² of the learned context and raw pixels."""
    labels = ["position", "velocity"]
    learned = [float(metrics["position_r2"]), float(metrics["velocity_r2"])]
    raw = [float(metrics["raw_position_r2"]), float(metrics["raw_velocity_r2"])]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - width / 2, learned, width, label="context encoder")
    ax.bar(x + width / 2, raw, width, label="raw pixels")
    ax.set_xticks(x, labels)
    ax.set_ylabel("ridge R²")
    ax.set_ylim(min(0.0, min(learned + raw) - 0.05), 1.0)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Train and evaluate the two-headed baseline."""
    dataset = generate_dataset(SEED, NUM_SAMPLES)
    train_sequence, train_time = pair_indices(dataset, split=0, k=K)
    val_sequence, val_time = pair_indices(dataset, split=1, k=K)
    test_sequence, test_time = pair_indices(dataset, split=2, k=K)

    val_context, val_pixels, val_state = fixed_batch(
        dataset, val_sequence, val_time, K, EVAL_PAIRS
    )
    test_context, test_pixels, test_state = fixed_batch(
        dataset, test_sequence, test_time, K, EVAL_PAIRS
    )

    print(f"frames: {dataset.frames.shape}")
    print(
        f"pairs: train={len(train_sequence):,} "
        f"val={len(val_sequence):,} test={len(test_sequence):,}"
    )
    print(
        f"context={val_context.shape} pixels={val_pixels.shape} "
        f"state={val_state.shape} k={K} steps={NUM_STEPS}"
    )

    state, history = train(
        dataset,
        train_sequence,
        train_time,
        val_context,
        val_pixels,
        val_state,
        jax.random.key(SEED),
    )

    test_pixel_mse, test_state_mse, test_persistence, pos_mae, vel_mae = (
        evaluate_baseline(
            state.params, test_context, test_pixels, test_state
        )
    )
    metrics = compute_probes(
        state, dataset, np.random.default_rng(SEED + 2)
    )
    metrics.update(
        {
            "val_pixel_mse": history[-1][2],
            "val_state_mse": history[-1][3],
            "val_persistence_mse": history[-1][4],
            "test_pixel_mse": test_pixel_mse,
            "test_state_mse": test_state_mse,
            "test_persistence_mse": test_persistence,
            "test_state_position_mae": pos_mae,
            "test_state_velocity_mae": vel_mae,
            "num_samples": NUM_SAMPLES,
            "k": K,
            "num_steps": NUM_STEPS,
            "state_weight": STATE_WEIGHT,
        }
    )

    write_config(RUN_DIR / "config.json")
    write_history(history, RUN_DIR / "metrics.csv")
    write_metrics(metrics, RUN_DIR / "probes.csv")
    plot_loss(history, RUN_DIR / "loss.png")
    plot_predictions(
        state.params, test_context, test_pixels, RUN_DIR / "predictions.png"
    )
    plot_probes(metrics, RUN_DIR / "probes.png")

    print(f"test pixel MSE: {test_pixel_mse:.6f}")
    print(f"test state MSE: {test_state_mse:.6f}")
    print(f"test persistence MSE: {test_persistence:.6f}")
    print(
        f"state MAE: position={pos_mae:.3f} px, "
        f"velocity={vel_mae:.3f} px/frame"
    )
    print(
        "probe R²: "
        f"position={metrics['position_r2']:.3f} "
        f"velocity={metrics['velocity_r2']:.3f} "
        f"raw position={metrics['raw_position_r2']:.3f} "
        f"raw velocity={metrics['raw_velocity_r2']:.3f}"
    )
    print(
        f"digit accuracy: {metrics['digit_accuracy']:.3f} "
        f"(raw {metrics['raw_digit_accuracy']:.3f})"
    )
    print(f"effective rank: {metrics['effective_rank']:.2f}")
    print(f"wrote {RUN_DIR}")


if __name__ == "__main__":
    main()
