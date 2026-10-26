"""Tests for variable-context phase-2 windows."""

from __future__ import annotations

import numpy as np

from toy_jepa.config import MNIST_DIR
from tiny_worlds.friction_mnist import generate_dataset
from toy_jepa.data import sample_batch


def test_batch_shapes_and_masks() -> None:
    dataset = generate_dataset(0, 20, mnist_dir=MNIST_DIR)
    batch = sample_batch(
        np.random.default_rng(0),
        dataset,
        split=0,
        batch_size=8,
        m_values=(2, 4),
        k_values=(1, 4),
        final_prob=0.5,
    )

    assert batch.context.shape == (8, 4, 64, 64)
    assert batch.mask.shape == (8, 4)
    assert batch.query.shape == (8, 2)
    assert batch.target_frame.shape == (8, 64, 64)
    assert batch.target_state.shape == (8, 8)
    assert batch.digit_label.shape == (8, 2)
    assert batch.bounce.shape == (8,)
    assert batch.horizon.shape == (8,)
    assert batch.final.shape == (8,)

    assert np.all(batch.mask.sum(axis=1) >= 2)
    assert np.all(batch.mask.sum(axis=1) <= 4)
    assert np.all(batch.context >= 0.0)
    assert np.all(batch.context <= 1.0)
    assert np.all((batch.bounce == 0.0) | (batch.bounce == 1.0))


def test_final_targets_are_at_rest() -> None:
    dataset = generate_dataset(0, 20, mnist_dir=MNIST_DIR)
    batch = sample_batch(
        np.random.default_rng(1),
        dataset,
        split=0,
        batch_size=4,
        m_values=(2, 4),
        k_values=(1, 4),
        final_prob=1.0,
    )

    assert np.all(batch.final)
    assert np.all(batch.target_state[:, 4:] == 0.0)
    assert np.all(batch.horizon >= 1)
