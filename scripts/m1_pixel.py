"""M1: pixel-only baseline plus the common post-hoc state readout.

The model is the pixel baseline: two context frames, a shared CNN encoder, a
fusion module, a predictor, and a pixel head. There is no supervised state
head. After training the model we freeze it, extract ``c_t`` and ``z_hat`` for
a large train/val/test set, and train the fixed-capacity MLP readout from
:mod:`toy_jepa.readout` on those features. Nothing is written except the run
directory: features stay in RAM.

Historical note: ``runs/m1_baseline/`` is the old two-headed run and is left
untouched. This script writes ``runs/m1_pixel/``.

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

from toy_jepa.constants import POSITION_SCALE, SEED, VELOCITY_SCALE
from toy_jepa.data import generate_dataset, pair_indices
from toy_jepa.experiment import (
    EvalRow,
    K,
    collapse_metrics,
    fixed_batch,
    readout_split,
    sample_batch,
)
from toy_jepa.metrics import (
    feature_variance,
    state_readout_metrics,
)
from toy_jepa.models.baseline import apply_baseline, init_baseline
from toy_jepa.optim import adamw_init
from toy_jepa.readout import apply_readout, train_readout
from toy_jepa.train import baseline_pixel_loss, train_step
from toy_jepa.types import (
    BaselineParams,
    Float32Array,
    Int32Array,
    MovingMNIST,
    TrainState,
)
from toy_jepa.viz import plot_dashboard

NUM_SAMPLES = 2_000
BATCH_SIZE = 64
NUM_STEPS = 1_500
EVAL_EVERY = 250
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
EMBED_DIM = 128
HIDDEN_DIM = 256

EVAL_PAIRS = 256
PROBE_PAIRS = 1_000
READOUT_TRAIN_PAIRS = 6_000
READOUT_VAL_PAIRS = 2_000
READOUT_TEST_PAIRS = 2_000
FEATURE_BATCH_SIZE = 256

READOUT_STEPS = 2_000
READOUT_BATCH_SIZE = 256
READOUT_LR = 1e-3
READOUT_WEIGHT_DECAY = 1e-4
READOUT_EVAL_EVERY = 50
READOUT_PATIENCE = 500

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m1_pixel"


def init_state(key: jax.Array) -> TrainState[BaselineParams]:
    """Initialize the baseline model and its AdamW moments."""
    params = init_baseline(
        key, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM
    )
    return TrainState(params=params, opt_state=adamw_init(params))


def encode_context(
    params: BaselineParams, context: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return frozen ``(c_t, z_hat)`` features for a context batch."""
    _, c_t, z_hat = apply_baseline(params, context)
    return c_t, z_hat


encode_context_jit = jax.jit(encode_context)


def extract_features(
    params: BaselineParams,
    context: jax.Array,
    *,
    batch_size: int = FEATURE_BATCH_SIZE,
) -> tuple[Float32Array, Float32Array]:
    """Extract NumPy ``c_t`` and ``z_hat`` arrays in bounded memory."""
    c_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    for start in range(0, len(context), batch_size):
        c_t, z_hat = encode_context_jit(
            params, context[start : start + batch_size]
        )
        c_chunks.append(np.asarray(c_t))
        z_chunks.append(np.asarray(z_hat))
    return np.concatenate(c_chunks), np.concatenate(z_chunks)


def evaluate_pixels(
    params: BaselineParams,
    context: jax.Array,
    target_pixels: jax.Array,
) -> tuple[float, float]:
    """Return pixel MSE and persistence MSE on a fixed batch."""
    pixels, _, _ = apply_baseline(params, context)
    pixel_mse = jnp.mean((pixels - target_pixels) ** 2)
    persistence = jnp.mean((context[:, 1] - target_pixels) ** 2)
    return float(pixel_mse), float(persistence)


def train(
    dataset: MovingMNIST,
    train_sequence: Int32Array,
    train_time: Int32Array,
    val_context: jax.Array,
    val_pixels: jax.Array,
    probe_context: jax.Array,
    key: jax.Array,
) -> tuple[TrainState[BaselineParams], list[EvalRow]]:
    """Train the pixel baseline and log loss/collapse diagnostics."""
    state = init_state(key)
    rng = np.random.default_rng(SEED + 1)

    @jax.jit
    def step(
        state: TrainState[BaselineParams],
        batch: tuple[jax.Array, ...],
    ) -> tuple[TrainState[BaselineParams], jax.Array]:
        return train_step(
            state,
            batch,
            baseline_pixel_loss,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    history: list[EvalRow] = []
    for i in range(NUM_STEPS):
        batch = sample_batch(
            rng,
            dataset,
            train_sequence,
            train_time,
            batch_size=BATCH_SIZE,
            k=K,
        )
        state, loss = step(state, batch)

        if i % EVAL_EVERY == 0 or i == NUM_STEPS - 1:
            val_pixel_mse, persistence = evaluate_pixels(
                state.params, val_context, val_pixels
            )
            probe_c, _ = extract_features(state.params, probe_context)
            collapse = collapse_metrics(probe_c)
            row = EvalRow(
                step=i,
                train_loss=float(loss),
                val_loss=val_pixel_mse,
                persistence=persistence,
                variance=collapse["feature_variance"],
                cosine=collapse["mean_pairwise_cosine"],
                rank=collapse["effective_rank"],
            )
            history.append(row)
            print(
                f"step {i:4d}  train {row.train_loss:.6f}  "
                f"val {row.val_loss:.6f}  "
                f"persistence {row.persistence:.6f}  "
                f"rank {row.rank:.2f}  "
                f"var {row.variance:.6f}  "
                f"cos {row.cosine:.3f}"
            )

    return state, history


def fit_readout(
    key: jax.Array,
    train_features: Float32Array,
    train_states: Float32Array,
    val_features: Float32Array,
    val_states: Float32Array,
    test_features: Float32Array,
    test_states: Float32Array,
    *,
    seed: int,
) -> tuple[dict[str, float], dict[str, float | int]]:
    """Train one frozen-feature readout and return test/validation metrics."""
    params, val_metrics = train_readout(
        key,
        train_features,
        train_states,
        val_features,
        val_states,
        num_steps=READOUT_STEPS,
        batch_size=READOUT_BATCH_SIZE,
        lr=READOUT_LR,
        weight_decay=READOUT_WEIGHT_DECAY,
        eval_every=READOUT_EVAL_EVERY,
        patience=READOUT_PATIENCE,
        seed=seed,
        position_scale=POSITION_SCALE,
        velocity_scale=VELOCITY_SCALE,
    )
    predictions = jax.jit(apply_readout)(params, test_features)
    test_metrics = state_readout_metrics(
        np.asarray(predictions),
        np.asarray(test_states),
        position_scale=POSITION_SCALE,
        velocity_scale=VELOCITY_SCALE,
    )
    return test_metrics, val_metrics


def write_history(history: list[EvalRow], path: Path) -> None:
    """Write one row per model evaluation to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(EvalRow._fields)
        writer.writerows(history)


def write_metrics(
    metrics: dict[str, float | int], path: Path
) -> None:
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
        "variant": "m1_pixel",
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
        "position_scale": POSITION_SCALE,
        "velocity_scale": VELOCITY_SCALE,
        "seed": SEED,
        "readout": {
            "architecture": [EMBED_DIM, HIDDEN_DIM, 8],
            "num_steps": READOUT_STEPS,
            "batch_size": READOUT_BATCH_SIZE,
            "learning_rate": READOUT_LR,
            "weight_decay": READOUT_WEIGHT_DECAY,
            "eval_every": READOUT_EVAL_EVERY,
            "patience": READOUT_PATIENCE,
            "train_pairs": READOUT_TRAIN_PAIRS,
            "val_pairs": READOUT_VAL_PAIRS,
            "test_pairs": READOUT_TEST_PAIRS,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def plot_loss(history: list[EvalRow], path: Path) -> None:
    """Plot the pixel training curve and persistence reference."""
    steps = [row.step for row in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, [row.train_loss for row in history], marker="o", markersize=3, label="train")
    ax.plot(steps, [row.val_loss for row in history], marker="o", markersize=3, label="val")
    ax.plot(
        steps,
        [row.persistence for row in history],
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


def main() -> None:
    """Train the pixel baseline, then train and evaluate the state readout."""
    dataset = generate_dataset(SEED, NUM_SAMPLES)
    train_sequence, train_time = pair_indices(dataset, split=0, k=K)
    val_sequence, val_time = pair_indices(dataset, split=1, k=K)
    test_sequence, test_time = pair_indices(dataset, split=2, k=K)

    val_context, val_pixels, _ = fixed_batch(
        dataset, val_sequence, val_time, num_pairs=EVAL_PAIRS, k=K
    )
    test_context, test_pixels, _ = fixed_batch(
        dataset, test_sequence, test_time, num_pairs=EVAL_PAIRS, k=K
    )
    probe_context, _, _ = fixed_batch(
        dataset, val_sequence, val_time, num_pairs=PROBE_PAIRS, k=K
    )

    print(f"frames: {dataset.frames.shape}")
    print(
        f"pairs: train={len(train_sequence):,} "
        f"val={len(val_sequence):,} test={len(test_sequence):,}"
    )
    print(
        f"context={val_context.shape} pixels={val_pixels.shape} "
        f"k={K} steps={NUM_STEPS}"
    )

    model_key, c_readout_key, z_readout_key = jax.random.split(
        jax.random.key(SEED), 3
    )
    state, history = train(
        dataset,
        train_sequence,
        train_time,
        val_context,
        val_pixels,
        probe_context,
        model_key,
    )

    test_pixel_mse, test_persistence = evaluate_pixels(
        state.params, test_context, test_pixels
    )

    readout_rng = np.random.default_rng(SEED + 2)
    train_context, train_states = readout_split(
        dataset, readout_rng, split=0, num_pairs=READOUT_TRAIN_PAIRS
    )
    val_readout_context, val_states = readout_split(
        dataset, readout_rng, split=1, num_pairs=READOUT_VAL_PAIRS
    )
    test_readout_context, test_states = readout_split(
        dataset, readout_rng, split=2, num_pairs=READOUT_TEST_PAIRS
    )

    print(
        f"readout sets: train={len(train_states):,} "
        f"val={len(val_states):,} test={len(test_states):,}"
    )
    train_c, train_z = extract_features(state.params, train_context)
    val_c, val_z = extract_features(state.params, val_readout_context)
    test_c, test_z = extract_features(state.params, test_readout_context)
    del train_context, val_readout_context, test_readout_context

    c_test, c_val = fit_readout(
        c_readout_key,
        train_c,
        np.asarray(train_states),
        val_c,
        np.asarray(val_states),
        test_c,
        np.asarray(test_states),
        seed=SEED + 3,
    )
    z_test, z_val = fit_readout(
        z_readout_key,
        train_z,
        np.asarray(train_states),
        val_z,
        np.asarray(val_states),
        test_z,
        np.asarray(test_states),
        seed=SEED + 4,
    )

    c_collapse = collapse_metrics(test_c)
    z_collapse = collapse_metrics(test_z)
    target_variance = feature_variance(np.asarray(test_states))

    metrics: dict[str, float | int] = {
        "test_pixel_mse": test_pixel_mse,
        "test_persistence_mse": test_persistence,
        "val_pixel_mse": history[-1].val_loss,
        "val_persistence_mse": history[-1].persistence,
        "target_variance": target_variance,
        "c_t_state_mse": c_test["state_mse"],
        "c_t_position_mae": c_test["position_mae"],
        "c_t_velocity_mae": c_test["velocity_mae"],
        "c_t_r2": c_test["r2"],
        "c_t_variance": c_collapse["feature_variance"],
        "c_t_cosine": c_collapse["mean_pairwise_cosine"],
        "c_t_rank": c_collapse["effective_rank"],
        "c_t_readout_best_val_mse": float(c_val["best_val_mse"]),
        "c_t_readout_best_step": int(c_val["best_step"]),
        "c_t_readout_steps_run": int(c_val["steps_run"]),
        "z_hat_state_mse": z_test["state_mse"],
        "z_hat_position_mae": z_test["position_mae"],
        "z_hat_velocity_mae": z_test["velocity_mae"],
        "z_hat_r2": z_test["r2"],
        "z_hat_variance": z_collapse["feature_variance"],
        "z_hat_cosine": z_collapse["mean_pairwise_cosine"],
        "z_hat_rank": z_collapse["effective_rank"],
        "z_hat_readout_best_val_mse": float(z_val["best_val_mse"]),
        "z_hat_readout_best_step": int(z_val["best_step"]),
        "z_hat_readout_steps_run": int(z_val["steps_run"]),
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "k": K,
    }

    write_config(RUN_DIR / "config.json")
    write_history(history, RUN_DIR / "metrics.csv")
    write_metrics(metrics, RUN_DIR / "probes.csv")
    plot_loss(history, RUN_DIR / "loss.png")
    plot_dashboard(
        history,
        {
            "c_t": c_test,
            "z_hat": z_test,
        },
        RUN_DIR / "dashboard.png",
        title="M1 pixel-only baseline + post-hoc state readout",
    )

    print(f"test pixel MSE: {test_pixel_mse:.6f}")
    print(f"test persistence MSE: {test_persistence:.6f}")
    for name, result in (("c_t", c_test), ("z_hat", z_test)):
        print(
            f"{name} readout: "
            f"MSE={result['state_mse']:.6f}  "
            f"pos MAE={result['position_mae']:.3f} px  "
            f"vel MAE={result['velocity_mae']:.3f} px/frame  "
            f"R²={result['r2']:.3f}"
        )
    print(
        "c_t collapse: "
        f"var={c_collapse['feature_variance']:.6f}  "
        f"cos={c_collapse['mean_pairwise_cosine']:.3f}  "
        f"rank={c_collapse['effective_rank']:.2f}  "
        f"target var={target_variance:.6f}"
    )
    print(f"wrote {RUN_DIR}")


if __name__ == "__main__":
    main()
