"""Shared batching, state normalization, and evaluation records."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .constants import POSITION_SCALE, VELOCITY_SCALE
from .data import make_context_target, pair_indices, state_at
from .metrics import effective_rank, feature_variance, mean_pairwise_cosine
from .types import Float32Array, Int32Array, MovingMNIST, UInt8Array

K = 4


def normalize_frames(frames: UInt8Array) -> jax.Array:
    return jnp.asarray(frames, dtype=jnp.float32) / 255.0


def normalize_state(position: Float32Array, velocity: Float32Array) -> jax.Array:
    positions = jnp.asarray(position, dtype=jnp.float32) / POSITION_SCALE
    velocities = jnp.asarray(velocity, dtype=jnp.float32) / VELOCITY_SCALE
    return jnp.concatenate([positions, velocities], axis=-1)


def make_batch(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    *,
    k: int = K,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(context, target_frame, target_state)`` for the pairs."""
    context, target_frame = make_context_target(
        dataset, sequence_idx, time_idx, k
    )
    position, velocity, _ = state_at(dataset, sequence_idx, time_idx + k)
    return (
        normalize_frames(context),
        normalize_frames(target_frame),
        normalize_state(position, velocity),
    )


def sample_indices(
    rng: np.random.Generator,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    count: int,
    *,
    replace: bool = False,
) -> tuple[Int32Array, Int32Array]:
    """Draw ``(sequence, time)`` pairs, without replacement by default."""
    if len(sequence_idx) == 0:
        raise ValueError("cannot sample from an empty pair set.")
    if count < 1:
        raise ValueError("count must be positive.")
    if not replace:
        count = min(count, len(sequence_idx))
    choice = rng.choice(len(sequence_idx), size=count, replace=replace)
    return sequence_idx[choice], time_idx[choice]


def sample_batch(
    rng: np.random.Generator, dataset: MovingMNIST,
    sequence_idx: Int32Array, time_idx: Int32Array, *,
    batch_size: int, k: int = K,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    choice = rng.integers(0, len(sequence_idx), size=batch_size)
    return make_batch(
        dataset, sequence_idx[choice], time_idx[choice], k=k
    )


def fixed_batch(
    dataset: MovingMNIST, sequence_idx: Int32Array,
    time_idx: Int32Array, *, num_pairs: int, k: int = K,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    count = min(num_pairs, len(sequence_idx))
    return make_batch(
        dataset, sequence_idx[:count], time_idx[:count], k=k
    )


def readout_split(
    dataset: MovingMNIST, rng: np.random.Generator,
    split: int, num_pairs: int, *, k: int = K,
) -> tuple[jax.Array, jax.Array]:
    """Sample readout contexts and normalized target states for one split."""
    sequence_idx, time_idx = pair_indices(dataset, split, k)
    sequence_idx, time_idx = sample_indices(
        rng, sequence_idx, time_idx, num_pairs
    )
    context, _, target_state = make_batch(
        dataset, sequence_idx, time_idx, k=k
    )
    return context, target_state


def collapse_metrics(features: Float32Array) -> dict[str, float]:
    return {
        "effective_rank": effective_rank(features),
        "feature_variance": feature_variance(features),
        "mean_pairwise_cosine": mean_pairwise_cosine(features),
    }


class EvalRow(NamedTuple):
    """One evaluation row shared by the collapse dashboards."""

    step: int
    train_loss: float
    val_loss: float
    persistence: float
    variance: float
    cosine: float
    rank: float


_COLLAPSE_NAMES = {"effective_rank": "rank", "feature_variance": "variance", "mean_pairwise_cosine": "cosine"}


def summarize_readouts(
    readouts: dict[
        str, tuple[dict[str, float], dict[str, float | int], dict[str, float]]
    ],
) -> dict[str, float | int]:
    """Flatten test, validation, and collapse metrics with a feature prefix."""
    summary: dict[str, float | int] = {}
    for name, (test, val, collapse) in readouts.items():
        summary.update({f"{name}_{key}": value for key, value in test.items()})
        summary.update(
            {f"{name}_{_COLLAPSE_NAMES[key]}": value for key, value in collapse.items()}
        )
        summary[f"{name}_readout_best_val_mse"] = float(val["best_val_mse"])
        summary[f"{name}_readout_best_step"] = int(val["best_step"])
        summary[f"{name}_readout_steps_run"] = int(val["steps_run"])
    return summary
