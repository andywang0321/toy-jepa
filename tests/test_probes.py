"""Tests for unordered digit-pair probes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from toy_jepa.probes import digit_accuracy, pair_index


def test_digit_accuracy_accepts_swapped_slots() -> None:
    labels = np.asarray([[2, 1], [3, 4]], dtype=np.int32)
    logits = np.zeros((2, 55), dtype=np.float32)
    logits[np.arange(2), np.asarray(pair_index(jnp.asarray(labels)))] = 10.0

    assert digit_accuracy(logits, labels) == 1.0
