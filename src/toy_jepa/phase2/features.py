"""Feature collection for phase-2 probes."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.types import MovingMNIST
from .data import WindowBatch, sample_batch

Apply = Callable[[Any, WindowBatch], tuple[Any, Any]]


def features(apply: Apply, params: Any, batch: WindowBatch) -> tuple[np.ndarray, np.ndarray]:
    c_t, z_hat = apply(params, batch)
    return np.asarray(c_t), np.asarray(z_hat)


def state_features(
    apply: Apply,
    params: Any,
    dataset: MovingMNIST,
    split: int,
    count: int,
    *,
    m_values: tuple[int, ...],
    k_values: tuple[int, ...],
    final_prob: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = sample_batch(
        rng,
        dataset,
        split,
        count,
        m_values=m_values,
        k_values=k_values,
        final_prob=final_prob,
    )
    c_t, z_hat = features(apply, params, batch)
    query = np.asarray(batch.query)
    return {
        "c_t": np.concatenate([c_t, query], axis=-1),
        "z_hat": np.concatenate([z_hat, query], axis=-1),
        "target": np.asarray(batch.target_state),
        "m": np.asarray(batch.mask.sum(axis=1)),
    }


def digit_features(
    apply: Apply,
    params: Any,
    dataset: MovingMNIST,
    split: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = sample_batch(rng, dataset, split, count, final_prob=0.5)
    c_t, z_hat = features(apply, params, batch)
    return c_t, z_hat, np.asarray(batch.digit_label)


def bounce_features(
    apply: Apply,
    params: Any,
    dataset: MovingMNIST,
    split: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = sample_batch(rng, dataset, split, count, final_prob=0.5)
    c_t, z_hat = features(apply, params, batch)
    query = np.asarray(batch.query)
    return (
        np.concatenate([c_t, query], axis=-1),
        np.concatenate([z_hat, query], axis=-1),
        np.asarray(batch.bounce),
    )
