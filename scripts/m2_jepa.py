"""M2: barebones JEPA on Moving MNIST.

The context and target use the same frame encoder, and there is no
stop-gradient. This is the control experiment: if the representation
collapses, that is the expected result.

Usage:
    uv run python scripts/m2_jepa.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple

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
from toy_jepa.models.cnn import apply_cnn
from toy_jepa.models.jepa_base import apply_jepa_base, init_jepa_base
from toy_jepa.optim import adamw_init
from toy_jepa.train import jepa_base_loss, train_step
from toy_jepa.types import (
    CNNParams,
    Float32Array,
    Int8Array,
    Int32Array,
    JEPABaseParams,
    MovingMNIST,
    TrainState,
    UInt8Array,
)

NUM_SAMPLES = 2_000
K = 4
BATCH_SIZE = 64
NUM_STEPS = 1_500
EVAL_EVERY = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EMBED_DIM = 128
HIDDEN_DIM = 256
EVAL_PAIRS = 256
PROBE_PAIRS = 1_000
FEATURE_RIDGE_ALPHA = 1.0
RAW_RIDGE_ALPHA = 100.0
RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m2_jepa"


class ProbeData(NamedTuple):
    """Fixed contexts and labels used for representation probes."""

    train_context: jax.Array
    test_context: jax.Array
    train_position: Float32Array
    test_position: Float32Array
    train_velocity: Float32Array
    test_velocity: Float32Array
    train_labels: Int8Array
    test_labels: Int8Array
    raw_train: Float32Array
    raw_test: Float32Array


class EvalRow(NamedTuple):
    """One row of the collapse dashboard."""

    step: int
    train_loss: float
    val_loss: float
    latent_persistence_mse: float
    effective_rank: float
    feature_variance: float
    mean_pairwise_cosine: float
    position_r2: float
    velocity_r2: float
    digit_accuracy: float


def _normalize_frames(frames: UInt8Array) -> jax.Array:
    """Scale uint8 frames to float32 in [0, 1]."""
    return jnp.asarray(frames, dtype=jnp.float32) / 255.0


def _batch_from_indices(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    context, target = make_context_target(
        dataset, sequence_idx, time_idx, k
    )
    return _normalize_frames(context), _normalize_frames(target)


def sample_batch(
    rng: np.random.Generator,
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample context frames and a target frame."""
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


def make_probe_data(
    dataset: MovingMNIST,
    rng: np.random.Generator,
) -> ProbeData:
    """Build the fixed train/test probe set."""
    train_context, train_position, train_velocity, train_labels = probe_batch(
        dataset, rng, split=0, k=K, num_pairs=PROBE_PAIRS
    )
    test_context, test_position, test_velocity, test_labels = probe_batch(
        dataset, rng, split=2, k=K, num_pairs=PROBE_PAIRS
    )
    raw_train = np.asarray(train_context[:, 1]).reshape(len(train_context), -1)
    raw_test = np.asarray(test_context[:, 1]).reshape(len(test_context), -1)
    return ProbeData(
        train_context=train_context,
        test_context=test_context,
        train_position=train_position,
        test_position=test_position,
        train_velocity=train_velocity,
        test_velocity=test_velocity,
        train_labels=train_labels,
        test_labels=test_labels,
        raw_train=raw_train,
        raw_test=raw_test,
    )


def init_state(key: jax.Array) -> TrainState[JEPABaseParams]:
    """Initialize the barebones JEPA model and its AdamW moments."""
    params = init_jepa_base(
        key, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    )
    return TrainState(params=params, opt_state=adamw_init(params))


def predict(
    params: JEPABaseParams,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the predicted latent and the fused context feature."""
    return apply_jepa_base(params, context)


predict_jit = jax.jit(predict)


def encode_frames(encoder: CNNParams, frames: jax.Array) -> jax.Array:
    """Encode a batch of frames with the shared frame encoder."""
    return apply_cnn(encoder, frames[..., None])


encode_frames_jit = jax.jit(encode_frames)


def evaluate_jepa(
    params: JEPABaseParams,
    context: jax.Array,
    target_frame: jax.Array,
) -> tuple[float, float]:
    """Return (latent JEPA loss, latent persistence loss)."""
    z_hat, _ = predict_jit(params, context)
    z_target = encode_frames_jit(params.encoder, target_frame)
    z_current = encode_frames_jit(params.encoder, context[:, 1])

    loss = jnp.mean((z_hat - z_target) ** 2)
    persistence = jnp.mean((z_current - z_target) ** 2)
    return float(loss), float(persistence)


def evaluate_probes(
    state: TrainState[JEPABaseParams],
    probes: ProbeData,
) -> dict[str, float]:
    """Measure collapse and linear readability of the context features."""
    _, train_features = predict_jit(state.params, probes.train_context)
    _, test_features = predict_jit(state.params, probes.test_context)
    train_features = np.asarray(train_features)
    test_features = np.asarray(test_features)

    return {
        "position_r2": ridge_r2(
            train_features,
            probes.train_position,
            test_features,
            probes.test_position,
            alpha=FEATURE_RIDGE_ALPHA,
        ),
        "velocity_r2": ridge_r2(
            train_features,
            probes.train_velocity,
            test_features,
            probes.test_velocity,
            alpha=FEATURE_RIDGE_ALPHA,
        ),
        "digit_accuracy": knn_accuracy(
            train_features,
            probes.train_labels,
            test_features,
            probes.test_labels,
        ),
        "effective_rank": effective_rank(test_features),
        "feature_variance": feature_variance(test_features),
        "mean_pairwise_cosine": mean_pairwise_cosine(test_features),
        "probe_pairs": float(len(test_features)),
    }


def raw_probe_metrics(probes: ProbeData) -> dict[str, float]:
    """Raw-pixel reference for the probes."""
    return {
        "raw_position_r2": ridge_r2(
            probes.raw_train,
            probes.train_position,
            probes.raw_test,
            probes.test_position,
            alpha=RAW_RIDGE_ALPHA,
        ),
        "raw_velocity_r2": ridge_r2(
            probes.raw_train,
            probes.train_velocity,
            probes.raw_test,
            probes.test_velocity,
            alpha=RAW_RIDGE_ALPHA,
        ),
        "raw_digit_accuracy": knn_accuracy(
            probes.raw_train,
            probes.train_labels,
            probes.raw_test,
            probes.test_labels,
        ),
    }


def train(
    dataset: MovingMNIST,
    train_sequence: Int32Array,
    train_time: Int32Array,
    val_context: jax.Array,
    val_target: jax.Array,
    probes: ProbeData,
    key: jax.Array,
) -> tuple[TrainState[JEPABaseParams], list[EvalRow]]:
    """Train barebones JEPA and record loss, collapse, and probe metrics."""
    state = init_state(key)
    rng = np.random.default_rng(SEED + 1)

    @jax.jit
    def step(
        state: TrainState[JEPABaseParams],
        batch: tuple[jax.Array, jax.Array],
    ) -> tuple[TrainState[JEPABaseParams], jax.Array]:
        return train_step(
            state,
            batch,
            jepa_base_loss,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    history: list[EvalRow] = []
    for i in range(NUM_STEPS):
        batch = sample_batch(
            rng, dataset, train_sequence, train_time, K, BATCH_SIZE
        )
        state, loss = step(state, batch)

        if i % EVAL_EVERY == 0 or i == NUM_STEPS - 1:
            val_loss, persistence = evaluate_jepa(
                state.params, val_context, val_target
            )
            probe_metrics = evaluate_probes(state, probes)
            row = EvalRow(
                step=i,
                train_loss=float(loss),
                val_loss=val_loss,
                latent_persistence_mse=persistence,
                effective_rank=probe_metrics["effective_rank"],
                feature_variance=probe_metrics["feature_variance"],
                mean_pairwise_cosine=probe_metrics[
                    "mean_pairwise_cosine"
                ],
                position_r2=probe_metrics["position_r2"],
                velocity_r2=probe_metrics["velocity_r2"],
                digit_accuracy=probe_metrics["digit_accuracy"],
            )
            history.append(row)
            print(
                f"step {i:4d}  train {row.train_loss:.6f}  "
                f"val {row.val_loss:.6f}  "
                f"rank {row.effective_rank:.2f}  "
                f"var {row.feature_variance:.6f}  "
                f"cos {row.mean_pairwise_cosine:.3f}  "
                f"pos {row.position_r2:.3f}  "
                f"vel {row.velocity_r2:.3f}  "
                f"digit {row.digit_accuracy:.3f}"
            )

    return state, history


def write_history(history: list[EvalRow], path: Path) -> None:
    """Write one row per evaluation to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(EvalRow._fields)
        writer.writerows(history)


def write_metrics(metrics: dict[str, float], path: Path) -> None:
    """Write final metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def write_config(path: Path) -> None:
    """Write the run configuration for reproducibility."""
    config = {
        "variant": "jepa_base",
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


def plot_collapse(
    history: list[EvalRow],
    raw_metrics: dict[str, float],
    path: Path,
) -> None:
    """Plot the loss, collapse diagnostics, and probes over training."""
    steps = [row.step for row in history]
    train_losses = [row.train_loss for row in history]
    val_losses = [row.val_loss for row in history]
    persistence = [row.latent_persistence_mse for row in history]
    ranks = [row.effective_rank for row in history]
    variances = [max(row.feature_variance, 1e-12) for row in history]
    cosines = [row.mean_pairwise_cosine for row in history]
    positions = [row.position_r2 for row in history]
    velocities = [row.velocity_r2 for row in history]
    digits = [row.digit_accuracy for row in history]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))

    axes[0, 0].plot(steps, train_losses, marker="o", markersize=3, label="train")
    axes[0, 0].plot(steps, val_losses, marker="o", markersize=3, label="val")
    axes[0, 0].plot(steps, persistence, linestyle="--", label="persistence")
    axes[0, 0].set_ylabel("latent MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, ranks, marker="o", markersize=3, color="tab:orange")
    axes[0, 1].set_ylabel("effective rank")
    axes[0, 1].set_ylim(bottom=0.0)

    axes[0, 2].plot(steps, variances, marker="o", markersize=3, color="tab:green")
    axes[0, 2].set_ylabel("feature variance")
    axes[0, 2].set_yscale("log")

    axes[1, 0].plot(steps, cosines, marker="o", markersize=3, color="tab:red")
    axes[1, 0].set_ylabel("mean pairwise cosine")
    axes[1, 0].set_ylim(0.0, 1.05)

    axes[1, 1].plot(steps, positions, marker="o", markersize=3, label="position")
    axes[1, 1].plot(steps, velocities, marker="o", markersize=3, label="velocity")
    axes[1, 1].axhline(
        raw_metrics["raw_position_r2"],
        color="tab:blue",
        linestyle=":",
        label="raw position",
    )
    axes[1, 1].axhline(
        raw_metrics["raw_velocity_r2"],
        color="tab:orange",
        linestyle=":",
        label="raw velocity",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.5)
    axes[1, 1].set_ylabel("ridge R²")
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].plot(steps, digits, marker="o", markersize=3, color="tab:purple")
    axes[1, 2].axhline(0.1, color="black", linestyle="--", label="chance")
    axes[1, 2].axhline(
        raw_metrics["raw_digit_accuracy"],
        color="tab:gray",
        linestyle=":",
        label="raw pixels",
    )
    axes[1, 2].set_ylabel("digit accuracy")
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].legend(fontsize=7)

    for axis in axes.flat:
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)

    fig.suptitle("M2 barebones JEPA: collapse dashboard", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_probes(metrics: dict[str, float], path: Path) -> None:
    """Compare learned context probes with raw pixels."""
    labels = ["position", "velocity"]
    learned = [metrics["position_r2"], metrics["velocity_r2"]]
    raw = [metrics["raw_position_r2"], metrics["raw_velocity_r2"]]

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
    """Train and evaluate barebones JEPA."""
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
    probes = make_probe_data(dataset, np.random.default_rng(SEED + 2))
    raw_metrics = raw_probe_metrics(probes)

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
        probes,
        jax.random.key(SEED),
    )

    test_loss, test_persistence = evaluate_jepa(
        state.params, test_context, test_target
    )
    final_probes = evaluate_probes(state, probes)
    metrics = {
        **raw_metrics,
        **final_probes,
        "val_loss": history[-1].val_loss,
        "val_latent_persistence_mse": history[-1].latent_persistence_mse,
        "test_loss": test_loss,
        "test_latent_persistence_mse": test_persistence,
        "num_samples": float(NUM_SAMPLES),
        "k": float(K),
        "num_steps": float(NUM_STEPS),
    }

    write_config(RUN_DIR / "config.json")
    write_history(history, RUN_DIR / "metrics.csv")
    write_metrics(metrics, RUN_DIR / "probes.csv")
    plot_collapse(history, raw_metrics, RUN_DIR / "collapse.png")
    plot_probes(metrics, RUN_DIR / "probes.png")

    print(f"test latent MSE: {test_loss:.6f}")
    print(f"test latent persistence: {test_persistence:.6f}")
    print(
        "final probes: "
        f"rank={metrics['effective_rank']:.2f} "
        f"var={metrics['feature_variance']:.6f} "
        f"cos={metrics['mean_pairwise_cosine']:.3f} "
        f"pos={metrics['position_r2']:.3f} "
        f"vel={metrics['velocity_r2']:.3f} "
        f"digit={metrics['digit_accuracy']:.3f}"
    )
    print(f"wrote {RUN_DIR}")


if __name__ == "__main__":
    main()
