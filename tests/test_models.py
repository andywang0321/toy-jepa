"""Tests for the phase-2 transformer and world models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from toy_jepa.data import WindowBatch
from toy_jepa.models import (
    apply_flow_cnn,
    apply_flow_mlp,
    apply_jepa,
    apply_pixel,
    flow_cnn_loss,
    flow_mlp_loss,
    init_flow_cnn_model,
    init_flow_mlp,
    init_jepa,
    init_pixel,
    jepa_loss,
    pixel_loss,
    sample_flow_cnn,
)


def _batch() -> WindowBatch:
    context = jnp.zeros((2, 3, 64, 64), dtype=jnp.float32)
    return WindowBatch(
        context=context,
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
        query=jnp.asarray([[0.1, 0.0], [1.0, 1.0]], dtype=jnp.float32),
        target_frame=jnp.zeros((2, 64, 64), dtype=jnp.float32),
        target_state=jnp.zeros((2, 8), dtype=jnp.float32),
        digit_label=jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32),
        bounce=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        horizon=jnp.asarray([4, 30], dtype=jnp.int32),
        final=jnp.asarray([False, True]),
    )


def test_pixel_forward_and_grad() -> None:
    params = init_pixel(jax.random.key(0), embed_dim=32, heads=4, layers=1, mlp_ratio=2, pixel_hidden=32)
    c_t, z_hat, frame = apply_pixel(params, _batch())

    assert c_t.shape == (2, 32)
    assert z_hat.shape == (2, 32)
    assert frame.shape == (2, 64, 64)
    assert np.isfinite(float(pixel_loss(params, _batch())))
    grads = jax.grad(pixel_loss)(params, _batch())
    assert np.isfinite(float(jax.tree.reduce(lambda a, b: a + jnp.sum(b), grads, 0.0)))


def test_jepa_forward_and_grad() -> None:
    params = init_jepa(jax.random.key(1), embed_dim=32, heads=4, layers=1, mlp_ratio=2)
    c_t, z_hat = apply_jepa(params, _batch())

    assert c_t.shape == (2, 32)
    assert z_hat.shape == (2, 32)
    assert np.isfinite(float(jepa_loss(params, _batch())))
    grads = jax.grad(jepa_loss)(params, _batch())
    assert np.isfinite(float(jax.tree.reduce(lambda a, b: a + jnp.sum(b), grads, 0.0)))


def test_flow_mlp_forward_and_grad() -> None:
    params = init_flow_mlp(jax.random.key(2), embed_dim=32, heads=4, layers=1, mlp_ratio=2, pixel_hidden=32)
    c_t, x_t, residual = apply_flow_mlp(params, _batch(), jax.random.key(3))

    assert c_t.shape == (2, 32)
    assert x_t.shape == (2, 64, 64)
    assert residual.shape == (2, 64, 64)
    assert np.isfinite(float(flow_mlp_loss(params, _batch(), jax.random.key(3))))
    grads = jax.grad(flow_mlp_loss, argnums=0)(params, _batch(), jax.random.key(3))
    assert np.isfinite(float(jax.tree.reduce(lambda a, b: a + jnp.sum(b), grads, 0.0)))


def test_flow_cnn_forward_and_grad() -> None:
    params = init_flow_cnn_model(jax.random.key(4), embed_dim=32, heads=4, layers=1, mlp_ratio=2)
    c_t, x_t, residual = apply_flow_cnn(params, _batch(), jax.random.key(5))

    assert c_t.shape == (2, 32)
    assert x_t.shape == (2, 64, 64)
    assert residual.shape == (2, 64, 64)
    assert sample_flow_cnn(params, _batch(), jax.random.key(6), steps=2).shape == (2, 64, 64)
    assert np.isfinite(float(flow_cnn_loss(params, _batch(), jax.random.key(5))))
    grads = jax.grad(flow_cnn_loss, argnums=0)(params, _batch(), jax.random.key(5))
    assert np.isfinite(float(jax.tree.reduce(lambda a, b: a + jnp.sum(b), grads, 0.0)))
