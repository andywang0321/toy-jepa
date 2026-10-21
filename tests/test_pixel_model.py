"""Tests for the pixel-future baseline."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.pixel import apply_pixel_model, init_pixel_model
from toy_jepa.optim import adamw_init
from toy_jepa.train import pixel_mse_loss, train_step
from toy_jepa.types import TrainState


def test_pixel_model_output_shapes() -> None:
    params = init_pixel_model(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)

    pixels, z_context, z_hat = apply_pixel_model(params, context)

    assert pixels.shape == (2, 64, 64)
    assert z_context.shape == (2, 64)
    assert z_hat.shape == (2, 64)
    assert jnp.all(jnp.isfinite(pixels))


def test_pixel_model_has_finite_gradients() -> None:
    params = init_pixel_model(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target = jnp.zeros((2, 64, 64), dtype=jnp.float32)

    def loss(p):
        pixels, _, _ = apply_pixel_model(p, context)
        return jnp.mean((pixels - target) ** 2)

    grads = jax.grad(loss)(params)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))


def test_pixel_train_step_updates_state() -> None:
    params = init_pixel_model(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    state = TrainState(params=params, opt_state=adamw_init(params))
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target = jnp.zeros((2, 64, 64), dtype=jnp.float32)

    next_state, loss = train_step(
        state,
        (context, target),
        pixel_mse_loss,
        lr=1e-3,
        weight_decay=0.0,
    )

    assert int(next_state.opt_state.t) == 1
    assert jnp.isfinite(loss)
    assert jax.tree.structure(next_state.params) == jax.tree.structure(params)
