"""Variable-context, variable-horizon friction-world windows."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..constants import POSITION_SCALE, VELOCITY_SCALE
from ..core.types import Float32Array, Int32Array, MovingMNIST
from .config import FINAL_PROB, HORIZON_SCALE, K_TRAIN, M_TRAIN


class WindowBatch(NamedTuple):
    context: jax.Array  # (B, max_m, 64, 64), [0, 1]
    mask: jax.Array  # (B, max_m)
    query: jax.Array  # (B, 2): k / HORIZON_SCALE, final flag
    target_frame: jax.Array  # (B, 64, 64), [0, 1]
    target_state: jax.Array  # (B, 8), normalized
    digit_label: jax.Array  # (B, 2)
    bounce: jax.Array  # (B,), bounce before target
    horizon: jax.Array  # (B,), actual frames ahead
    final: jax.Array  # (B,)


def normalize_state(position: Float32Array, velocity: Float32Array) -> jax.Array:
    positions = jnp.asarray(position, dtype=jnp.float32) / POSITION_SCALE
    velocities = jnp.asarray(velocity, dtype=jnp.float32) / VELOCITY_SCALE
    return jnp.concatenate([positions, velocities], axis=-1)


def bounce_between(
    velocity: Float32Array,
    start: int,
    end: int,
) -> float:
    """Whether any digit reverses direction between two frames."""
    window = velocity[start : end + 1]
    signs = np.sign(window)
    changes = (signs[1:] != signs[:-1]) & (signs[1:] != 0.0) & (signs[:-1] != 0.0)
    return float(np.any(changes))


def make_batch(
    dataset: MovingMNIST,
    sequences: Int32Array,
    times: Int32Array,
    ms: Int32Array,
    ks: Int32Array,
    final: np.ndarray,
) -> WindowBatch:
    """Build padded context windows for explicit query descriptors."""
    batch = len(sequences)
    max_m = int(ms.max())
    context = np.zeros((batch, max_m, *dataset.frames.shape[2:]), dtype=np.float32)
    mask = np.zeros((batch, max_m), dtype=bool)
    target_frame = np.empty((batch, *dataset.frames.shape[2:]), dtype=np.float32)
    target_state = np.empty((batch, 8), dtype=np.float32)
    digit_label = np.empty((batch, 2), dtype=np.int32)
    bounce = np.empty(batch, dtype=np.float32)
    horizon = np.empty(batch, dtype=np.int32)

    for row, (sequence, time, m, k, is_final) in enumerate(zip(sequences, times, ms, ks, final)):
        context[row, :m] = dataset.frames[sequence, time - m + 1 : time + 1]
        mask[row, :m] = True
        target_time = int(dataset.stop_time[sequence]) if is_final else int(time + k)
        target_frame[row] = dataset.frames[sequence, target_time]
        if is_final:
            target_state[row, :4] = dataset.final_position[sequence].reshape(-1) / POSITION_SCALE
            target_state[row, 4:] = 0.0
        else:
            target_state[row] = normalize_state(
                dataset.position[sequence, target_time],
                dataset.velocity[sequence, target_time],
            ).reshape(-1)
        digit_label[row] = dataset.digit_label[sequence]
        bounce[row] = bounce_between(dataset.velocity[sequence], int(time), target_time)
        horizon[row] = target_time - int(time)

    query = np.stack(
        [horizon.astype(np.float32) / HORIZON_SCALE, final.astype(np.float32)],
        axis=-1,
    )
    return WindowBatch(
        context=jnp.asarray(context / 255.0),
        mask=jnp.asarray(mask),
        query=jnp.asarray(query),
        target_frame=jnp.asarray(target_frame / 255.0),
        target_state=jnp.asarray(target_state),
        digit_label=jnp.asarray(digit_label),
        bounce=jnp.asarray(bounce),
        horizon=jnp.asarray(horizon),
        final=jnp.asarray(final),
    )


def _sample_window(
    rng: np.random.Generator,
    dataset: MovingMNIST,
    sequences: Int32Array,
    m_values: tuple[int, ...],
    k_values: tuple[int, ...],
    final_prob: float,
) -> tuple[int, int, int, int, bool]:
    """Sample one valid (sequence, time, m, k, final) tuple."""
    for _ in range(100):
        sequence = int(rng.choice(sequences))
        m = int(rng.choice(m_values))
        stop = int(dataset.stop_time[sequence])
        if stop < m:
            continue
        is_final = bool(rng.random() < final_prob)
        if is_final:
            if stop == m - 1:
                continue
            time = int(rng.integers(m - 1, stop))
            return sequence, time, m, stop - time, True
        k = int(rng.choice(k_values))
        if stop - k < m - 1:
            continue
        time = int(rng.integers(m - 1, stop - k + 1))
        return sequence, time, m, k, False
    raise RuntimeError("could not sample a valid phase-2 window.")


def sample_batch(
    rng: np.random.Generator,
    dataset: MovingMNIST,
    split: int,
    batch_size: int,
    *,
    m_values: tuple[int, ...] = M_TRAIN,
    k_values: tuple[int, ...] = K_TRAIN,
    final_prob: float = FINAL_PROB,
) -> WindowBatch:
    """Sample a training/evaluation batch from one split."""
    sequences = np.flatnonzero(dataset.split == split).astype(np.int32)
    if len(sequences) == 0:
        raise ValueError(f"split {split} is empty.")
    rows = [_sample_window(rng, dataset, sequences, m_values, k_values, final_prob) for _ in range(batch_size)]
    sequence = np.asarray([row[0] for row in rows], dtype=np.int32)
    time = np.asarray([row[1] for row in rows], dtype=np.int32)
    m = np.asarray([row[2] for row in rows], dtype=np.int32)
    k = np.asarray([row[3] for row in rows], dtype=np.int32)
    final = np.asarray([row[4] for row in rows], dtype=bool)
    return make_batch(dataset, sequence, time, m, k, final)
