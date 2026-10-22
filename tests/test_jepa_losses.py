"""Tests for the stop-grad, EMA, and SIGReg JEPA losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.jepa_base import init_jepa_base
from toy_jepa.optim import adamw_init
from toy_jepa.train import (
    ema_train_step,
    jepa_ema_loss,
    jepa_ema_sigreg_loss,
    jepa_stopgrad_loss,
)
from toy_jepa.types import JEPAEMAState, TrainState


def _batch() -> tuple[jax.Array, jax.Array, jax.Array]:
    context = jax.random.uniform(
        jax.random.key(0), (2, 2, 64, 64), dtype=jnp.float32
    )
    target_frame = jax.random.uniform(
        jax.random.key(1), (2, 64, 64), dtype=jnp.float32
    )
    target_state = jnp.zeros((2, 8), dtype=jnp.float32)
    return context, target_frame, target_state


def test_stopgrad_loss_is_finite() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )

    loss = jepa_stopgrad_loss(params, _batch())

    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_ema_loss_is_finite() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )

    loss = jepa_ema_loss(params, params.encoder, _batch())

    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_ema_sigreg_loss_is_finite() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )

    loss = jepa_ema_sigreg_loss(
        params,
        params.encoder,
        _batch(),
        key=jax.random.key(1),
        weight=0.1,
    )

    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_ema_target_receives_no_gradient() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )
    batch = _batch()

    grads = jax.grad(
        lambda target_encoder: jepa_ema_loss(params, target_encoder, batch)
    )(params.encoder)

    for grad in jax.tree.leaves(grads):
        assert jnp.all(grad == 0.0)


def test_ema_train_step_increments_optimizer_step() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )
    state = JEPAEMAState(
        train_state=TrainState(
            params=params, opt_state=adamw_init(params)
        ),
        target_encoder=params.encoder,
    )

    next_state, loss = ema_train_step(
        state,
        _batch(),
        jepa_ema_loss,
        lr=1e-3,
        weight_decay=0.0,
        decay=0.99,
    )

    assert int(next_state.train_state.opt_state.t) == 1
    assert jnp.isfinite(loss)
