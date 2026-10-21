"""Tests for probe metrics."""

from __future__ import annotations

import numpy as np

from toy_jepa.metrics import effective_rank, knn_accuracy, ridge_r2


def test_ridge_r2_recovers_a_linear_map() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 8)).astype(np.float32)
    targets = features[:, :4] @ np.diag([1.0, 2.0, -1.0, 0.5])
    train = slice(0, 150)
    test = slice(150, 200)

    score = ridge_r2(
        features[train], targets[train], features[test], targets[test], alpha=1e-6
    )
    assert score > 0.99


def test_knn_accuracy_on_separable_features() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(100, 4)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int8).reshape(-1, 1)

    accuracy = knn_accuracy(
        features[:70], labels[:70], features[70:], labels[70:], k=5
    )
    assert accuracy > 0.9


def test_effective_rank_is_bounded() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(100, 16)).astype(np.float32)

    rank = effective_rank(features)
    assert 1.0 <= rank <= 16.0
