"""Tests for the shared experiment batching and collapse helpers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from toy_jepa.constants import SEED
from toy_jepa.data import generate_dataset, pair_indices, state_at
from toy_jepa.experiment import (
    K,
    collapse_metrics,
    fixed_batch,
    make_batch,
    normalize_frames,
    normalize_state,
    readout_split,
    sample_batch,
    sample_indices,
    summarize_readouts,
)
from toy_jepa.metrics import (
    effective_rank,
    feature_variance,
    mean_pairwise_cosine,
)
from toy_jepa.types import MovingMNIST


@lru_cache(maxsize=1)
def _dataset() -> MovingMNIST:
    return generate_dataset(SEED, 20)


def test_make_batch_matches_data_helpers() -> None:
    dataset = _dataset()
    sequence, time = pair_indices(dataset, split=0, k=K)
    sequence, time = sequence[:4], time[:4]

    context, target, state = make_batch(dataset, sequence, time, k=K)

    assert context.shape == (4, 2, 64, 64)
    assert target.shape == (4, 64, 64)
    assert state.shape == (4, 8)
    assert float(context.min()) >= 0.0
    assert float(context.max()) <= 1.0
    assert float(target.max()) <= 1.0

    expected_target = normalize_frames(dataset.frames[sequence, time + K])
    np.testing.assert_allclose(np.asarray(target), np.asarray(expected_target))

    position, velocity, _ = state_at(dataset, sequence, time + K)
    expected_state = normalize_state(position, velocity)
    np.testing.assert_allclose(np.asarray(state), np.asarray(expected_state))


def test_sampling_and_fixed_batches() -> None:
    dataset = _dataset()
    sequence, time = pair_indices(dataset, split=0, k=K)

    sampled_sequence, sampled_time = sample_indices(
        np.random.default_rng(0), sequence, time, 10
    )
    repeat_sequence, repeat_time = sample_indices(
        np.random.default_rng(0), sequence, time, 10
    )
    assert len(sampled_sequence) == 10
    assert np.array_equal(sampled_sequence, repeat_sequence)
    assert np.array_equal(sampled_time, repeat_time)

    context, target, state = sample_batch(
        np.random.default_rng(1),
        dataset,
        sequence,
        time,
        batch_size=8,
        k=K,
    )
    assert context.shape == (8, 2, 64, 64)
    assert target.shape == (8, 64, 64)
    assert state.shape == (8, 8)

    fixed = fixed_batch(dataset, sequence, time, num_pairs=5, k=K)
    manual = make_batch(dataset, sequence[:5], time[:5], k=K)
    for fixed_array, manual_array in zip(fixed, manual):
        np.testing.assert_allclose(
            np.asarray(fixed_array), np.asarray(manual_array)
        )


def test_readout_split_is_deterministic() -> None:
    dataset = _dataset()

    context_a, state_a = readout_split(
        dataset, np.random.default_rng(0), split=0, num_pairs=30
    )
    context_b, state_b = readout_split(
        dataset, np.random.default_rng(0), split=0, num_pairs=30
    )

    assert context_a.shape == (30, 2, 64, 64)
    assert state_a.shape == (30, 8)
    np.testing.assert_array_equal(np.asarray(context_a), np.asarray(context_b))
    np.testing.assert_array_equal(np.asarray(state_a), np.asarray(state_b))


def test_collapse_metrics_match_metric_functions() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(50, 8)).astype(np.float32)

    metrics = collapse_metrics(features)

    assert metrics["effective_rank"] == pytest.approx(effective_rank(features))
    assert metrics["feature_variance"] == pytest.approx(
        feature_variance(features)
    )
    assert metrics["mean_pairwise_cosine"] == pytest.approx(
        mean_pairwise_cosine(features)
    )


def test_summarize_readouts_prefixes_each_feature_set() -> None:
    readouts = {
        "c_t": (
            {"state_mse": 1.0, "position_mae": 2.0, "velocity_mae": 3.0, "r2": 0.5},
            {"best_val_mse": 4.0, "best_step": 5, "steps_run": 6},
            {
                "effective_rank": 7.0,
                "feature_variance": 8.0,
                "mean_pairwise_cosine": 9.0,
            },
        )
    }

    summary = summarize_readouts(readouts)

    assert summary["c_t_state_mse"] == 1.0
    assert summary["c_t_r2"] == 0.5
    assert summary["c_t_rank"] == 7.0
    assert summary["c_t_variance"] == 8.0
    assert summary["c_t_cosine"] == 9.0
    assert summary["c_t_readout_best_val_mse"] == 4.0
    assert summary["c_t_readout_best_step"] == 5
    assert summary["c_t_readout_steps_run"] == 6
