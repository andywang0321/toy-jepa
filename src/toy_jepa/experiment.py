"""Shared experiment helpers for the M1-M5 comparison.

All arms use the same data protocol: two context frames ``(x_{t-1}, x_t)``,
a target frame ``x_{t+k}``, and the normalized 8-d state at target time
``t+k``. The helpers here build those batches and summarize a frozen feature
matrix; they do not train any particular model.
"""

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
    """Scale uint8 frames to float32 in ``[0, 1]``."""
    return jnp.asarray(frames, dtype=jnp.float32) / 255.0


def normalize_state(
    position: Float32Array,
    velocity: Float32Array,
    *,
    position_scale: float = POSITION_SCALE,
    velocity_scale: float = VELOCITY_SCALE,
) -> jax.Array:
    """Scale both digits' ``(x, y, vx, vy)`` to comparable ranges."""
    positions = jnp.asarray(position, dtype=jnp.float32) / position_scale
    velocities = jnp.asarray(velocity, dtype=jnp.float32) / velocity_scale
    return jnp.concatenate([positions, velocities], axis=-1)


def make_batch(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    *,
    k: int = K,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build ``(context, target_frame, target_state)`` for the given pairs.

    Shapes are ``(B, 2, 64, 64)``, ``(B, 64, 64)``, and ``(B, 8)``. The target
    state is the state at ``t + k``, normalized with the shared scales.
    """
    context, target_frame = make_context_target(
        dataset, sequence_idx, time_idx, k
    )
    position, velocity, _ = state_at(
        dataset, sequence_idx, time_idx + k
    )
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
    if not replace:
        count = min(count, len(sequence_idx))
    if count < 1:
        raise ValueError("count must be positive.")
    choice = rng.choice(len(sequence_idx), size=count, replace=replace)
    return sequence_idx[choice], time_idx[choice]


def sample_batch(
    rng: np.random.Generator,
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    *,
    batch_size: int,
    k: int = K,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample a training batch with replacement."""
    choice = rng.integers(0, len(sequence_idx), size=batch_size)
    return make_batch(
        dataset, sequence_idx[choice], time_idx[choice], k=k
    )


def fixed_batch(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    *,
    num_pairs: int,
    k: int = K,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build a deterministic batch from the first ``num_pairs`` pairs."""
    count = min(num_pairs, len(sequence_idx))
    return make_batch(
        dataset, sequence_idx[:count], time_idx[:count], k=k
    )


def readout_split(
    dataset: MovingMNIST,
    rng: np.random.Generator,
    split: int,
    num_pairs: int,
    *,
    k: int = K,
) -> tuple[jax.Array, jax.Array]:
    """Sample readout contexts and normalized target states for one split.

    Every arm should create one ``np.random.default_rng(SEED + 2)`` and call
    this helper in train/val/test order so all arms read out from the exact
    same context pairs.
    """
    sequence_idx, time_idx = pair_indices(dataset, split, k)
    sequence_idx, time_idx = sample_indices(
        rng, sequence_idx, time_idx, num_pairs
    )
    context, target_pixels, target_state = make_batch(
        dataset, sequence_idx, time_idx, k=k
    )
    del target_pixels
    return context, target_state


def collapse_metrics(features: Float32Array) -> dict[str, float]:
    """Rank, variance, and cosine diagnostics for a feature matrix."""
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
