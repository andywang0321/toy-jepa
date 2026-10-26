"""Smoke tests for frozen-c_t generative heads."""

from __future__ import annotations

import jax
import numpy as np

from toy_jepa.jepa_heads import (
    sample_flow_cnn_head,
    sample_vae_head,
    train_flow_cnn_head,
    train_vae_head,
)


def _data():
    key = jax.random.key(0)
    c_t = jax.random.normal(key, (8, 128))
    query = jax.random.normal(jax.random.key(1), (8, 128))
    target = jax.random.uniform(jax.random.key(2), (8, 64, 64))
    return c_t, query, target


def test_flow_cnn_head_shapes() -> None:
    c_t, query, target = _data()
    head = train_flow_cnn_head(c_t, query, target, jax.random.key(3), steps=2, batch_size=4)
    frame = sample_flow_cnn_head(head, c_t[:2], query[:2], jax.random.key(4), steps=2)
    assert frame.shape == (2, 64, 64)
    assert np.all(np.asarray(frame) >= 0.0)
    assert np.all(np.asarray(frame) <= 1.0)


def test_vae_head_shapes() -> None:
    c_t, query, target = _data()
    head = train_vae_head(c_t, query, target, jax.random.key(5), steps=2, batch_size=4)
    frame = sample_vae_head(head, c_t[:2], query[:2], jax.random.key(6))
    assert frame.shape == (2, 64, 64)
    assert np.all(np.asarray(frame) >= 0.0)
    assert np.all(np.asarray(frame) <= 1.0)
