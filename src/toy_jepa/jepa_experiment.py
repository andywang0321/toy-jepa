"""Shared runner for the M2-M5 JEPA arms.

All four arms use the same encoder, fusion, predictor, data protocol, probe set,
and fixed-capacity post-hoc state readout. The only differences are the target
gradient path and the optional SIGReg term:

- ``base``: shared encoder, no stop-gradient (M2);
- ``stopgrad``: stop-gradient on the shared-encoder target (M3);
- ``ema``: stop-gradient target from an EMA encoder (M4);
- ``sigreg``: EMA target plus SIGReg on ``c_t`` (M5).

The state readout is trained after the model on frozen online ``c_t``/``z_hat``
features, exactly as in M1. Features stay in RAM; only plots, CSVs, and JSON
are written.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from .constants import POSITION_SCALE, SEED, VELOCITY_SCALE
from .data import generate_dataset, pair_indices
from .experiment import (
    K,
    EvalRow,
    collapse_metrics,
    fixed_batch,
    readout_split,
    sample_batch,
)
from .metrics import feature_variance, state_readout_metrics
from .models.cnn import apply_cnn
from .models.jepa_base import apply_jepa_base, init_jepa_base
from .optim import adamw_init
from .readout import apply_readout, train_readout
from .train import (
    ema_train_step,
    jepa_base_loss,
    jepa_ema_loss,
    jepa_ema_sigreg_loss,
    jepa_stopgrad_loss,
    train_step,
)
from .types import (
    CNNParams,
    Float32Array,
    Int32Array,
    JEPABaseParams,
    JEPAEMAState,
    MovingMNIST,
    TrainState,
)
from .viz import plot_dashboard

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

VARIANTS = ("base", "stopgrad", "ema", "sigreg")
TITLES = {
    "base": "M2 barebones JEPA + common state readout",
    "stopgrad": "M3 stop-gradient JEPA + common state readout",
    "ema": "M4 EMA JEPA + common state readout",
    "sigreg": "M5 EMA + SIGReg JEPA + common state readout",
}


class JEPAConfig(NamedTuple):
    """One JEPA arm: variant, output directory, and shared constants."""

    variant: str
    run_dir: Path
    num_samples: int = 2_000
    k: int = K
    batch_size: int = 64
    num_steps: int = 1_500
    eval_every: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    embed_dim: int = 128
    hidden_dim: int = 256
    ema_decay: float = 0.99
    sigreg_weight: float = 0.1


Batch = tuple[jax.Array, jax.Array, jax.Array]
StateT = TypeVar("StateT")

predict_jit = jax.jit(apply_jepa_base)


def _encode_frames(encoder: CNNParams, frames: jax.Array) -> jax.Array:
    """Encode a batch of frames with one encoder."""
    return apply_cnn(encoder, frames[..., None])


encode_frames_jit = jax.jit(_encode_frames)


def _evaluate_loss(
    params: JEPABaseParams,
    target_encoder: CNNParams,
    context: jax.Array,
    target_frame: jax.Array,
) -> tuple[float, float, float]:
    """Return latent loss, online persistence, and target variance."""
    z_hat, _ = predict_jit(params, context)
    z_target = encode_frames_jit(target_encoder, target_frame)
    z_current = encode_frames_jit(params.encoder, context[:, 1])
    z_future = encode_frames_jit(params.encoder, target_frame)

    loss = jnp.mean((z_hat - z_target) ** 2)
    persistence = jnp.mean((z_current - z_future) ** 2)
    target_variance = feature_variance(np.asarray(z_target))
    return float(loss), float(persistence), target_variance


def _evaluate_arm(
    params: JEPABaseParams,
    target_encoder: CNNParams,
    val_context: jax.Array,
    val_target: jax.Array,
    probe_context: jax.Array,
    probe_target: jax.Array,
    *,
    step: int,
    train_loss: float,
) -> tuple[EvalRow, float]:
    """Evaluate loss/persistence and collapse diagnostics on fixed probes."""
    val_loss, persistence, _ = _evaluate_loss(
        params, target_encoder, val_context, val_target
    )
    _, c_t = predict_jit(params, probe_context)
    collapse = collapse_metrics(np.asarray(c_t))
    probe_target_features = encode_frames_jit(target_encoder, probe_target)
    target_variance = feature_variance(np.asarray(probe_target_features))

    row = EvalRow(
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        persistence=persistence,
        variance=collapse["feature_variance"],
        cosine=collapse["mean_pairwise_cosine"],
        rank=collapse["effective_rank"],
    )
    return row, target_variance


def _train_loop(
    config: JEPAConfig,
    dataset: MovingMNIST,
    train_sequence: Int32Array,
    train_time: Int32Array,
    val_context: jax.Array,
    val_target: jax.Array,
    probe_context: jax.Array,
    probe_target: jax.Array,
    state: StateT,
    step_fn: Callable[[StateT, Batch], tuple[StateT, jax.Array]],
    online_params: Callable[[StateT], JEPABaseParams],
    target_encoder: Callable[[StateT], CNNParams],
) -> tuple[StateT, list[EvalRow], float]:
    """Run one arm's training loop and log shared evaluation rows."""
    rng = np.random.default_rng(SEED + 1)
    history: list[EvalRow] = []
    target_variance = 0.0

    for i in range(config.num_steps):
        batch = sample_batch(
            rng,
            dataset,
            train_sequence,
            train_time,
            batch_size=config.batch_size,
            k=config.k,
        )
        state, loss = step_fn(state, batch)

        if i % config.eval_every == 0 or i == config.num_steps - 1:
            row, target_variance = _evaluate_arm(
                online_params(state),
                target_encoder(state),
                val_context,
                val_target,
                probe_context,
                probe_target,
                step=i,
                train_loss=float(loss),
            )
            history.append(row)
            print(
                f"step {i:4d}  train {row.train_loss:.6f}  "
                f"val {row.val_loss:.6f}  "
                f"persistence {row.persistence:.6f}  "
                f"rank {row.rank:.2f}  "
                f"var {row.variance:.6f}  "
                f"cos {row.cosine:.3f}  "
                f"target var {target_variance:.6f}"
            )

    return state, history, target_variance


def _extract_features(
    params: JEPABaseParams,
    context: jax.Array,
    *,
    batch_size: int = FEATURE_BATCH_SIZE,
) -> tuple[Float32Array, Float32Array]:
    """Extract frozen NumPy ``c_t`` and ``z_hat`` features in chunks."""
    c_chunks: list[np.ndarray] = []
    z_chunks: list[np.ndarray] = []
    for start in range(0, len(context), batch_size):
        z_hat, c_t = predict_jit(
            params, context[start : start + batch_size]
        )
        c_chunks.append(np.asarray(c_t))
        z_chunks.append(np.asarray(z_hat))
    return np.concatenate(c_chunks), np.concatenate(z_chunks)


def _fit_readout(
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


def _write_history(history: list[EvalRow], path: Path) -> None:
    """Write one evaluation row per model assessment to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(EvalRow._fields)
        writer.writerows(history)


def _write_metrics(
    metrics: dict[str, float | int], path: Path
) -> None:
    """Write final metrics as key/value CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def _write_config(config: JEPAConfig, path: Path) -> None:
    """Write the run configuration for reproducibility."""
    payload = {
        "variant": config.variant,
        "num_samples": config.num_samples,
        "k": config.k,
        "batch_size": config.batch_size,
        "num_steps": config.num_steps,
        "eval_every": config.eval_every,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "ema_decay": config.ema_decay,
        "sigreg_weight": config.sigreg_weight,
        "eval_pairs": EVAL_PAIRS,
        "probe_pairs": PROBE_PAIRS,
        "position_scale": POSITION_SCALE,
        "velocity_scale": VELOCITY_SCALE,
        "seed": SEED,
        "sigreg_key_seed": SEED + 5,
        "readout": {
            "architecture": [
                config.embed_dim,
                config.hidden_dim,
                8,
            ],
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
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _validate(config: JEPAConfig) -> None:
    """Reject unsupported arms and nonsensical loop settings."""
    if config.variant not in VARIANTS:
        raise ValueError(
            f"variant must be one of {VARIANTS}, got {config.variant!r}."
        )
    if config.num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if config.eval_every < 1:
        raise ValueError("eval_every must be positive.")
    if config.num_samples < 1:
        raise ValueError("num_samples must be positive.")


def run_jepa(config: JEPAConfig) -> None:
    """Train one JEPA arm, then fit and evaluate the common state readout."""
    _validate(config)
    dataset = generate_dataset(SEED, config.num_samples)
    train_sequence, train_time = pair_indices(dataset, split=0, k=config.k)
    val_sequence, val_time = pair_indices(dataset, split=1, k=config.k)
    test_sequence, test_time = pair_indices(dataset, split=2, k=config.k)

    val_context, val_target, _ = fixed_batch(
        dataset, val_sequence, val_time, num_pairs=EVAL_PAIRS, k=config.k
    )
    test_context, test_target, _ = fixed_batch(
        dataset, test_sequence, test_time, num_pairs=EVAL_PAIRS, k=config.k
    )
    probe_context, probe_target, _ = fixed_batch(
        dataset, val_sequence, val_time, num_pairs=PROBE_PAIRS, k=config.k
    )

    print(f"arm: {config.variant}")
    print(f"frames: {dataset.frames.shape}")
    print(
        f"pairs: train={len(train_sequence):,} "
        f"val={len(val_sequence):,} test={len(test_sequence):,}"
    )
    print(
        f"context={val_context.shape} target={val_target.shape} "
        f"k={config.k} steps={config.num_steps}"
    )

    model_key, c_readout_key, z_readout_key = jax.random.split(
        jax.random.key(SEED), 3
    )
    params = init_jepa_base(
        model_key, embed_dim=config.embed_dim, hidden_dim=config.hidden_dim
    )

    if config.variant in ("base", "stopgrad"):
        online_loss = (
            jepa_base_loss
            if config.variant == "base"
            else jepa_stopgrad_loss
        )
        state_online = TrainState(
            params=params, opt_state=adamw_init(params)
        )

        @jax.jit
        def step_online(
            state: TrainState[JEPABaseParams], batch: Batch
        ) -> tuple[TrainState[JEPABaseParams], jax.Array]:
            return train_step(
                state,
                (batch[0], batch[1]),
                online_loss,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        state_online, history, val_target_variance = _train_loop(
            config,
            dataset,
            train_sequence,
            train_time,
            val_context,
            val_target,
            probe_context,
            probe_target,
            state_online,
            step_online,
            lambda state: state.params,
            lambda state: state.params.encoder,
        )
        online = state_online.params
        target_encoder = state_online.params.encoder
    else:
        state_ema = JEPAEMAState(
            train_state=TrainState(
                params=params, opt_state=adamw_init(params)
            ),
            target_encoder=params.encoder,
        )
        if config.variant == "ema":
            ema_loss = jepa_ema_loss
        else:
            ema_loss = partial(
                jepa_ema_sigreg_loss,
                key=jax.random.key(SEED + 5),
                weight=config.sigreg_weight,
            )

        @jax.jit
        def step_ema(
            state: JEPAEMAState, batch: Batch
        ) -> tuple[JEPAEMAState, jax.Array]:
            return ema_train_step(
                state,
                batch,
                ema_loss,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                decay=config.ema_decay,
            )

        state_ema, history, val_target_variance = _train_loop(
            config,
            dataset,
            train_sequence,
            train_time,
            val_context,
            val_target,
            probe_context,
            probe_target,
            state_ema,
            step_ema,
            lambda state: state.train_state.params,
            lambda state: state.target_encoder,
        )
        online = state_ema.train_state.params
        target_encoder = state_ema.target_encoder

    test_loss, test_persistence, target_variance = _evaluate_loss(
        online, target_encoder, test_context, test_target
    )

    readout_rng = np.random.default_rng(SEED + 2)
    train_context, train_states = readout_split(
        dataset,
        readout_rng,
        split=0,
        num_pairs=READOUT_TRAIN_PAIRS,
        k=config.k,
    )
    val_readout_context, val_states = readout_split(
        dataset,
        readout_rng,
        split=1,
        num_pairs=READOUT_VAL_PAIRS,
        k=config.k,
    )
    test_readout_context, test_states = readout_split(
        dataset,
        readout_rng,
        split=2,
        num_pairs=READOUT_TEST_PAIRS,
        k=config.k,
    )
    print(
        f"readout sets: train={len(train_states):,} "
        f"val={len(val_states):,} test={len(test_states):,}"
    )

    train_c, train_z = _extract_features(online, train_context)
    val_c, val_z = _extract_features(online, val_readout_context)
    test_c, test_z = _extract_features(online, test_readout_context)
    del train_context, val_readout_context, test_readout_context

    c_test, c_val = _fit_readout(
        c_readout_key,
        train_c,
        np.asarray(train_states),
        val_c,
        np.asarray(val_states),
        test_c,
        np.asarray(test_states),
        seed=SEED + 3,
    )
    z_test, z_val = _fit_readout(
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

    metrics: dict[str, float | int] = {
        "test_loss": test_loss,
        "test_latent_persistence_mse": test_persistence,
        "val_loss": history[-1].val_loss,
        "val_latent_persistence_mse": history[-1].persistence,
        "target_variance": target_variance,
        "val_target_variance": val_target_variance,
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
        "num_samples": config.num_samples,
        "num_steps": config.num_steps,
        "k": config.k,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "ema_decay": config.ema_decay,
        "sigreg_weight": config.sigreg_weight,
    }

    config.run_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config, config.run_dir / "config.json")
    _write_history(history, config.run_dir / "metrics.csv")
    _write_metrics(metrics, config.run_dir / "probes.csv")
    plot_dashboard(
        history,
        {"c_t": c_test, "z_hat": z_test},
        config.run_dir / "dashboard.png",
        title=TITLES[config.variant],
    )

    print(f"test latent MSE: {test_loss:.6f}")
    print(f"test latent persistence: {test_persistence:.6f}")
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
    print(f"wrote {config.run_dir}")
