"""Tests for the pixel-only baseline model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.baseline import apply_baseline, init_baseline
from toy_jepa.optim import adamw_init
from toy_jepa.train import baseline_pixel_loss, train_step
from toy_jepa.types import TrainState


def test_baseline_output_shapes() -> None:
    params = init_baseline(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)

    pixels, c_t, z_hat = apply_baseline(params, context)

    assert pixels.shape == (2, 64, 64)
    assert c_t.shape == (2, 64)
    assert z_hat.shape == (2, 64)
    assert jnp.all(jnp.isfinite(pixels))
    assert jnp.all(jnp.isfinite(c_t))
    assert jnp.all(jnp.isfinite(z_hat))


def test_baseline_has_no_state_head() -> None:
    params = init_baseline(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )

    assert params._fields == ("encoder", "fusion", "predictor", "pixel_head")


def test_baseline_has_finite_gradients() -> None:
    params = init_baseline(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target_pixels = jnp.zeros((2, 64, 64), dtype=jnp.float32)

    def loss(p):
        pixels, _, _ = apply_baseline(p, context)
        return jnp.mean((pixels - target_pixels) ** 2)

    grads = jax.grad(loss)(params)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))


def test_baseline_train_step_updates_state() -> None:
    params = init_baseline(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    state = TrainState(params=params, opt_state=adamw_init(params))
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target_pixels = jnp.zeros((2, 64, 64), dtype=jnp.float32)
    target_state = jnp.zeros((2, 8), dtype=jnp.float32)

    # The target state is ignored by the pixel-only loss; keep the canonical
    # three-entry experiment batch.
    batch = (context, target_pixels, target_state)
    next_state, loss = train_step(
        state,
        batch,
        baseline_pixel_loss,
        lr=1e-3,
        weight_decay=0.0,
    )

    assert int(next_state.opt_state.t) == 1
    assert jnp.isfinite(loss)
    assert jax.tree.structure(next_state.params) == jax.tree.structure(params)
