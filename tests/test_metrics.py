"""Tests for probe metrics."""

from __future__ import annotations

import numpy as np
import pytest

from toy_jepa.constants import POSITION_SCALE, VELOCITY_SCALE
from toy_jepa.evaluation.metrics import (
    effective_rank,
    knn_accuracy,
    ridge_r2,
    state_readout_metrics,
)


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

    accuracy = knn_accuracy(features[:70], labels[:70], features[70:], labels[70:], k=5)
    assert accuracy > 0.9


def test_effective_rank_is_bounded() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(100, 16)).astype(np.float32)

    rank = effective_rank(features)
    assert 1.0 <= rank <= 16.0


def test_state_readout_metrics_perfect_prediction() -> None:
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(20, 8)).astype(np.float32)

    metrics = state_readout_metrics(targets, targets)

    assert metrics["state_mse"] == 0.0
    assert metrics["position_mae"] == 0.0
    assert metrics["velocity_mae"] == 0.0
    assert metrics["r2"] == pytest.approx(1.0)


def test_state_readout_metrics_use_pixel_units() -> None:
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(20, 8)).astype(np.float32)
    predictions = targets.copy()
    predictions[:, :4] += 1.0 / POSITION_SCALE
    predictions[:, 4:] += 1.0 / VELOCITY_SCALE

    metrics = state_readout_metrics(predictions, targets)

    expected_mse = np.mean(
        [(1.0 / POSITION_SCALE) ** 2] * 4 + [(1.0 / VELOCITY_SCALE) ** 2] * 4
    )
    assert metrics["state_mse"] == pytest.approx(expected_mse, rel=1e-5)
    assert metrics["position_mae"] == pytest.approx(1.0)
    assert metrics["velocity_mae"] == pytest.approx(1.0)
    assert metrics["r2"] < 1.0
