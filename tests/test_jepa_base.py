"""Tests for the barebones JEPA model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.jepa_base import apply_jepa_base, init_jepa_base
from toy_jepa.optim import adamw_init
from toy_jepa.train import jepa_base_loss, train_step
from toy_jepa.types import TrainState


def test_jepa_base_output_shapes() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)

    z_hat, c_t = apply_jepa_base(params, context)

    assert z_hat.shape == (2, 64)
    assert c_t.shape == (2, 64)
    assert jnp.all(jnp.isfinite(z_hat))
    assert jnp.all(jnp.isfinite(c_t))


def test_jepa_base_loss_has_finite_gradients() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target_frame = jnp.zeros((2, 64, 64), dtype=jnp.float32)

    def loss(p):
        return jepa_base_loss(p, (context, target_frame))

    grads = jax.grad(loss)(params)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))


def test_jepa_base_train_step_updates_state() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=64, hidden_dim=64
    )
    state = TrainState(params=params, opt_state=adamw_init(params))
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target_frame = jnp.zeros((2, 64, 64), dtype=jnp.float32)

    def loss_fn(p, batch):
        return jepa_base_loss(p, batch)

    next_state, loss = train_step(
        state,
        (context, target_frame),
        loss_fn,
        lr=1e-3,
        weight_decay=0.0,
    )

    assert int(next_state.opt_state.t) == 1
    assert jnp.isfinite(loss)
    assert jax.tree.structure(next_state.params) == jax.tree.structure(params)
