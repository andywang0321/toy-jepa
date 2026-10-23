"""Tests for context/target pair construction."""

from __future__ import annotations

import numpy as np

from toy_jepa.constants import SEED
from toy_jepa.data.dataset import (
    generate_dataset,
    make_context_target,
    pair_indices,
    state_at,
)


def test_pair_shapes_and_ranges() -> None:
    dataset = generate_dataset(SEED, 20)
    sequence, time = pair_indices(dataset, split=0, k=4)

    assert len(sequence) > 0
    assert time.min() == 1
    assert time.max() == dataset.frames.shape[1] - 4 - 1

    context, target = make_context_target(dataset, sequence[:3], time[:3], 4)
    position, velocity, labels = state_at(dataset, sequence[:3], time[:3])

    assert context.shape == (3, 2, 64, 64)
    assert target.shape == (3, 64, 64)
    assert position.shape == (3, 4)
    assert velocity.shape == (3, 4)
    assert labels.shape == (3, 2)
    assert np.all(time + 4 < dataset.frames.shape[1])
