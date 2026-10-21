"""M1: pixel-future baseline on Moving MNIST.

Context is two frames, ``(x_{t-1}, x_t)``; the model predicts ``x_{t+k}``.
This is the generative contrast for the JEPA variants: same frame encoder,
fusion, and predictor, but a pixel head and pixel MSE.

Usage:
    uv run python scripts/m1_pixel.py
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

from toy_jepa.constants import SEED
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
from toy_jepa.models.pixel import apply_pixel_model, init_pixel_model
from toy_jepa.optim import adamw_init
from toy_jepa.train import pixel_mse_loss, train_step
from toy_jepa.types import (
    Float32Array,
    Int8Array,
    Int32Array,
    MovingMNIST,
    PixelParams,
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
EMBED_DIM = 128
HIDDEN_DIM = 256
EVAL_PAIRS = 256
PROBE_PAIRS = 1_000
FEATURE_RIDGE_ALPHA = 1.0
RAW_RIDGE_ALPHA = 100.0
RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m1_pixel"


def _normalize_frames(frames: UInt8Array) -> jax.Array:
    """Scale uint8 frames to float32 in [0, 1]."""
    return jnp.asarray(frames, dtype=jnp.float32) / 255.0


def _batch_from_indices(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    context, target = make_context_target(dataset, sequence_idx, time_idx, k)
    return _normalize_frames(context), _normalize_frames(target)


def sample_batch(
    rng: np.random.Generator,
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample a random training batch of context/target pairs."""
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
) -> tuple[jax.Array, jax.Array]:
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


def init_state(key: jax.Array) -> TrainState[PixelParams]:
    """Initialize the pixel model and its AdamW moments."""
    params = init_pixel_model(
        key, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    )
    return TrainState(params=params, opt_state=adamw_init(params))


def predict(
    params: PixelParams, context: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return predicted pixels and the context representation."""
    pixels, z_context, _ = apply_pixel_model(params, context)
    return pixels, z_context


predict_jit = jax.jit(predict)


def evaluate_pixel(
    params: PixelParams,
    context: jax.Array,
    target: jax.Array,
) -> tuple[float, float]:
    """Return (pixel MSE, persistence MSE)."""
    pixels, _ = predict_jit(params, context)
    mse = jnp.mean((pixels - target) ** 2)
    # Persistence: assume nothing changes, so x_{t+k} is just the current x_t.
    persistence = jnp.mean((context[:, 1] - target) ** 2)
    return float(mse), float(persistence)


def train(
    dataset: MovingMNIST,
    train_sequence: Int32Array,
    train_time: Int32Array,
    val_context: jax.Array,
    val_target: jax.Array,
    key: jax.Array,
) -> tuple[TrainState[PixelParams], list[tuple[int, float, float, float]]]:
    """Train the pixel baseline and log train/val losses."""
    state = init_state(key)
    rng = np.random.default_rng(SEED + 1)

    @jax.jit
    def step(
        state: TrainState[PixelParams],
        batch: tuple[jax.Array, jax.Array],
    ) -> tuple[TrainState[PixelParams], jax.Array]:
        return train_step(
            state,
            batch,
            pixel_mse_loss,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    history: list[tuple[int, float, float, float]] = []
    for i in range(NUM_STEPS):
        context, target = sample_batch(
            rng, dataset, train_sequence, train_time, K, BATCH_SIZE
        )
        state, loss = step(state, (context, target))

        if i % EVAL_EVERY == 0 or i == NUM_STEPS - 1:
            val_mse, persistence = evaluate_pixel(
                state.params, val_context, val_target
            )
            row = (i, float(loss), val_mse, persistence)
            history.append(row)
            print(
                f"step {i:4d}  train {row[1]:.6f}  "
                f"val {val_mse:.6f}  persistence {persistence:.6f}"
            )

    return state, history


def compute_probes(
    state: TrainState[PixelParams],
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

    _, train_features = predict_jit(state.params, train_context)
    _, test_features = predict_jit(state.params, test_context)
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
    history: list[tuple[int, float, float, float]], path: Path
) -> None:
    """Write train/val metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "train_mse", "val_mse", "persistence_mse"])
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
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "eval_pairs": EVAL_PAIRS,
        "probe_pairs": PROBE_PAIRS,
        "feature_ridge_alpha": FEATURE_RIDGE_ALPHA,
        "raw_ridge_alpha": RAW_RIDGE_ALPHA,
        "seed": SEED,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def plot_loss(
    history: list[tuple[int, float, float, float]], path: Path
) -> None:
    """Plot training, validation, and persistence MSE on a log scale."""
    steps, train_losses, val_losses, persistence = zip(*history)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, train_losses, marker="o", markersize=3, label="train")
    ax.plot(steps, val_losses, marker="o", markersize=3, label="val")
    ax.plot(
        steps,
        persistence,
        linestyle="--",
        label="persistence",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("pixel MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(
    params: PixelParams,
    context: jax.Array,
    target: jax.Array,
    path: Path,
) -> None:
    """Plot current frame, target, prediction, and persistence."""
    count = min(4, len(context))
    pixels, _ = predict_jit(params, context[:count])
    pixels_np = np.asarray(pixels)
    context_np = np.asarray(context)
    target_np = np.asarray(target)

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

    fig.suptitle("Pixel-future predictions (columns are test samples)", fontsize=11)
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
    """Train and evaluate the pixel-future baseline."""
    dataset = generate_dataset(SEED, NUM_SAMPLES)
    train_sequence, train_time = pair_indices(dataset, split=0, k=K)
    val_sequence, val_time = pair_indices(dataset, split=1, k=K)
    test_sequence, test_time = pair_indices(dataset, split=2, k=K)

    val_context, val_target = fixed_batch(
        dataset, val_sequence, val_time, K, EVAL_PAIRS
    )
    test_context, test_target = fixed_batch(
        dataset, test_sequence, test_time, K, EVAL_PAIRS
    )

    print(f"frames: {dataset.frames.shape}")
    print(
        f"pairs: train={len(train_sequence):,} "
        f"val={len(val_sequence):,} test={len(test_sequence):,}"
    )
    print(
        f"context={val_context.shape} target={val_target.shape} "
        f"k={K} steps={NUM_STEPS}"
    )

    state, history = train(
        dataset,
        train_sequence,
        train_time,
        val_context,
        val_target,
        jax.random.key(SEED),
    )

    test_mse, test_persistence = evaluate_pixel(
        state.params, test_context, test_target
    )
    metrics = compute_probes(
        state, dataset, np.random.default_rng(SEED + 2)
    )
    metrics.update(
        {
            "val_mse": history[-1][2],
            "val_persistence_mse": history[-1][3],
            "test_mse": test_mse,
            "test_persistence_mse": test_persistence,
            "num_samples": NUM_SAMPLES,
            "k": K,
            "num_steps": NUM_STEPS,
        }
    )

    write_config(RUN_DIR / "config.json")
    write_history(history, RUN_DIR / "metrics.csv")
    write_metrics(metrics, RUN_DIR / "probes.csv")
    plot_loss(history, RUN_DIR / "loss.png")
    plot_predictions(
        state.params, test_context, test_target, RUN_DIR / "predictions.png"
    )
    plot_probes(metrics, RUN_DIR / "probes.png")

    print(f"test pixel MSE: {test_mse:.6f}")
    print(f"test persistence MSE: {test_persistence:.6f}")
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
