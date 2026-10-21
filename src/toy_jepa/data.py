"""Moving MNIST generation.

``generate_dataset(seed, num_samples)`` is the only entry point for obtaining
data. It loads MNIST through :mod:`toy_jepa.mnist_utils` and returns a fixed,
labeled in-memory pool. Nothing is written to disk.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    SEED, NUM_SAMPLES,
    CANVAS_SIZE,
    DIGIT_SIZE,
    NUM_FRAMES,
    SPEED_RANGE,
    TRAIN_FRACTION,
    VAL_FRACTION,
)
from .mnist_utils import load_mnist
from .types import Float32Array, Int8Array, Int32Array, MovingMNIST, UInt8Array


def _reflect(
    position: Float32Array,
    velocity: Float32Array,
    max_position: float,
) -> tuple[Float32Array, Float32Array]:
    """Reflect positions and velocities at the walls [0, max_position]."""
    below = position < 0.0
    above = position > max_position
    position = np.where(below, -position, position)
    position = np.where(above, 2.0 * max_position - position, position)
    velocity = np.where(below | above, -velocity, velocity)
    return position.astype(np.float32), velocity.astype(np.float32)


def _render_sequences(
    images: UInt8Array,
    digit_index: Int32Array,
    position: Float32Array,
    *,
    canvas_size: int,
    digit_size: int,
) -> UInt8Array:
    """Render frames by stamping each digit at its rounded top-left position."""
    num_sequences, num_frames, _, _ = position.shape
    frames = np.zeros(
        (num_sequences, num_frames, canvas_size, canvas_size), dtype=np.uint8
    )
    patches = images[digit_index]  # (N, 2, digit_size, digit_size)
    xs = np.rint(position[..., 0]).astype(np.int32)
    ys = np.rint(position[..., 1]).astype(np.int32)
    max_xy = canvas_size - digit_size

    for sequence in range(num_sequences):
        for frame in range(num_frames):
            for digit in range(2):
                x = min(max(int(xs[sequence, frame, digit]), 0), max_xy)
                y = min(max(int(ys[sequence, frame, digit]), 0), max_xy)
                patch = patches[sequence, digit]
                window = frames[
                    sequence, frame, y : y + digit_size, x : x + digit_size
                ]
                frames[
                    sequence, frame, y : y + digit_size, x : x + digit_size
                ] = np.maximum(window, patch)
    return frames


def _sample_digit_indices(
    rng: np.random.Generator,
    *,
    num_sequences: int,
    num_images: int,
) -> Int32Array:
    """Sample two MNIST image indices per sequence."""
    return rng.integers(0, num_images, size=(num_sequences, 2), dtype=np.int32)


def _sample_motion(
    rng: np.random.Generator,
    *,
    num_sequences: int,
    num_frames: int,
    max_position: float,
    speed_range: tuple[float, float],
) -> tuple[Float32Array, Float32Array]:
    """Sample initial positions and velocities, then simulate the bounces."""
    position = rng.uniform(
        0.0, max_position, size=(num_sequences, 2, 2)
    ).astype(np.float32)
    speed = rng.uniform(
        speed_range[0], speed_range[1], size=(num_sequences, 2, 1)
    ).astype(np.float32)
    angle = rng.uniform(
        0.0, 2.0 * np.pi, size=(num_sequences, 2, 1)
    ).astype(np.float32)
    velocity = (
        np.concatenate([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)
        * speed
    )

    positions = np.empty((num_sequences, num_frames, 2, 2), dtype=np.float32)
    velocities = np.empty_like(positions)
    current_position = position
    current_velocity = velocity
    for frame in range(num_frames):
        positions[:, frame] = current_position
        velocities[:, frame] = current_velocity
        current_position = current_position + current_velocity
        current_position, current_velocity = _reflect(
            current_position, current_velocity, max_position
        )
    return positions, velocities


def _make_split(
    num_sequences: int,
    *,
    train_fraction: float,
    val_fraction: float,
) -> Int8Array:
    """Assign split ids in contiguous blocks before shuffling."""
    num_train = int(num_sequences * train_fraction)
    num_val = int(num_sequences * val_fraction)
    split = np.full(num_sequences, 2, dtype=np.int8)
    split[:num_train] = 0
    split[num_train : num_train + num_val] = 1
    return split


def generate_dataset(seed: int = SEED, num_samples: int = NUM_SAMPLES) -> MovingMNIST:
    """Generate a fixed, labeled Moving MNIST pool.

    Each sample contains two MNIST digits with constant speed and elastic
    bounces off the canvas walls. Positions are top-left pixel coordinates.

    Args:
        seed: Seed for the NumPy generator.
        num_samples: Number of sequences in the pool.

    Returns:
        A ``MovingMNIST`` dataset with a train/val/test split.
    """
    images, labels = load_mnist()
    if images.shape[1:] != (DIGIT_SIZE, DIGIT_SIZE):
        raise ValueError(
            f"MNIST images must be {DIGIT_SIZE}x{DIGIT_SIZE}, got {images.shape[1:]}."
        )
    if labels.shape != (images.shape[0],):
        raise ValueError(
            f"labels must have shape ({images.shape[0]},), got {labels.shape}."
        )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    rng = np.random.default_rng(seed)
    max_position = float(CANVAS_SIZE - DIGIT_SIZE)

    digit_index = _sample_digit_indices(
        rng, num_sequences=num_samples, num_images=images.shape[0]
    )
    digit_label = labels[digit_index].astype(np.int8)
    positions, velocities = _sample_motion(
        rng,
        num_sequences=num_samples,
        num_frames=NUM_FRAMES,
        max_position=max_position,
        speed_range=SPEED_RANGE,
    )
    frames = _render_sequences(
        images,
        digit_index,
        positions,
        canvas_size=CANVAS_SIZE,
        digit_size=DIGIT_SIZE,
    )
    split = _make_split(
        num_samples,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
    )

    order = rng.permutation(num_samples)
    return MovingMNIST(
        frames=frames[order],
        digit_label=digit_label[order],
        digit_index=digit_index[order],
        position=positions[order],
        velocity=velocities[order],
        split=split[order],
    )


def pair_indices(
    dataset: MovingMNIST,
    split: int,
    k: int,
) -> tuple[Int32Array, Int32Array]:
    """Return all ``(sequence, time)`` pairs for a context window.

    The context is ``(x_{t-1}, x_t)`` and the target is ``x_{t+k}``. ``t``
    ranges over ``[1, T - k)`` so the target always exists.
    """
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
        [
            dataset.frames[sequence_idx, time_idx - 1],
            dataset.frames[sequence_idx, time_idx],
        ],
        axis=1,
    )
    target = dataset.frames[sequence_idx, time_idx + k]
    return context, target


def state_at(
    dataset: MovingMNIST,
    sequence_idx: Int32Array,
    time_idx: Int32Array,
) -> tuple[Float32Array, Float32Array, Int8Array]:
    """Return current position, velocity, and digit labels for each pair."""
    position = dataset.position[sequence_idx, time_idx].reshape(-1, 4)
    velocity = dataset.velocity[sequence_idx, time_idx].reshape(-1, 4)
    labels = dataset.digit_label[sequence_idx]
    return position, velocity, labels
