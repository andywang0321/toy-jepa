"""Friction Moving MNIST access and context/target helpers."""

from __future__ import annotations

import numpy as np
from tiny_worlds.friction_mnist import generate_dataset as generate_friction_dataset

from ..constants import NUM_SAMPLES, SEED
from ..core.types import Float32Array, Int8Array, Int32Array, MovingMNIST, UInt8Array


def generate_dataset(
    seed: int = SEED,
    num_samples: int = NUM_SAMPLES,
) -> MovingMNIST:
    """Generate a friction Moving MNIST pool in memory."""
    return generate_friction_dataset(seed, num_samples)


def pair_indices(
    dataset: MovingMNIST,
    split: int,
    k: int,
) -> tuple[Int32Array, Int32Array]:
    """Return all ``(sequence, time)`` pairs for a two-frame context."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    sequences = np.flatnonzero(dataset.split == split)
    times = np.arange(1, dataset.frames.shape[1] - k)
    if len(sequences) == 0 or len(times) == 0:
        raise ValueError(f"No pairs for split={split}, k={k}.")
    sequence_grid, time_grid = np.meshgrid(sequences, times, indexing="ij")
    return (
        sequence_grid.ravel().astype(np.int32),
        time_grid.ravel().astype(np.int32),
    )


def make_context_target(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
    k: int,
) -> tuple[UInt8Array, UInt8Array]:
    """Return context ``(x_{t-1}, x_t)`` and target ``x_{t+k}``."""
    context = np.stack(
        [dataset.frames[sequence_idx, time_idx - 1], dataset.frames[sequence_idx, time_idx]],
        axis=1,
    )
    return context, dataset.frames[sequence_idx, time_idx + k]


def state_at(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
) -> tuple[Float32Array, Float32Array, Int8Array]:
    """Return current position, velocity, and digit labels."""
    position = dataset.position[sequence_idx, time_idx].reshape(-1, 4)
    velocity = dataset.velocity[sequence_idx, time_idx].reshape(-1, 4)
    labels = dataset.digit_label[sequence_idx]
    return position, velocity, labels
