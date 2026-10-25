"""Tests for permutation-invariant digit accuracy."""

from __future__ import annotations

import numpy as np

from toy_jepa.phase2.readout import digit_accuracy


def test_digit_accuracy_accepts_swapped_slots() -> None:
    logits = np.zeros((2, 20), dtype=np.float32)
    logits[0].reshape(2, 10)[0, 1] = 10.0
    logits[0].reshape(2, 10)[1, 2] = 10.0
    logits[1].reshape(2, 10)[0, 3] = 10.0
    logits[1].reshape(2, 10)[1, 4] = 10.0
    labels = np.asarray([[2, 1], [3, 4]], dtype=np.int32)

    assert digit_accuracy(logits, labels) == 1.0
