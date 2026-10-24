"""Tests for the barebones JEPA model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.jepa_base import (
    JEPABaseParams,
    apply_jepa_base,
    init_jepa_base,
    jepa_base_loss,
)
from toy_jepa.core.optim import adamw_init
from toy_jepa.core.train import train_step
from toy_jepa.core.types import TrainState


def _batch() -> tuple[jax.Array, jax.Array]:
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target_frame = jnp.zeros((2, 64, 64), dtype=jnp.float32)
    return context, target_frame


def test_jepa_base_output_shapes() -> None:
    params = init_jepa_base(jax.random.key(0), embed_dim=64, hidden_dim=64)
    context, _ = _batch()

    z_hat, c_t = apply_jepa_base(params, context)

    assert isinstance(params, JEPABaseParams)
    assert z_hat.shape == (2, 64)
    assert c_t.shape == (2, 64)
    assert jnp.all(jnp.isfinite(z_hat))
    assert jnp.all(jnp.isfinite(c_t))


def test_jepa_base_loss_has_finite_gradients() -> None:
    params = init_jepa_base(jax.random.key(0), embed_dim=64, hidden_dim=64)
    batch = _batch()

    grads = jax.grad(lambda p: jepa_base_loss(p, batch))(params)

    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))


def test_jepa_base_train_step_updates_state() -> None:
    params = init_jepa_base(jax.random.key(0), embed_dim=64, hidden_dim=64)
    state = TrainState(params=params, opt_state=adamw_init(params))

    next_state, loss = train_step(state, _batch(), jepa_base_loss, lr=1e-3, weight_decay=0.0)

    assert int(next_state.opt_state.t) == 1
    assert jnp.isfinite(loss)
    assert jax.tree.structure(next_state.params) == jax.tree.structure(params)
