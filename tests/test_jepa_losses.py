"""Tests for the stop-grad, EMA, and SIGReg JEPA losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.cnn import apply_cnn
from toy_jepa.models.jepa_base import init_jepa_base, jepa_base_loss
from toy_jepa.models.jepa_ema import init_jepa_ema, jepa_ema_loss
from toy_jepa.models.jepa_sigreg import (
    apply_jepa_sigreg,
    init_jepa_sigreg,
    jepa_sigreg_loss,
)
from toy_jepa.models.jepa_stopgrad import (
    init_jepa_stopgrad,
    jepa_stopgrad_loss,
)
from toy_jepa.optim import adamw_init
from toy_jepa.train import ema_train_step
from toy_jepa.types import EMAState, TrainState


def _batch() -> tuple[jax.Array, jax.Array]:
    context = jax.random.uniform(
        jax.random.key(0), (2, 2, 64, 64), dtype=jnp.float32
    )
    target_frame = jax.random.uniform(
        jax.random.key(1), (2, 64, 64), dtype=jnp.float32
    )
    return context, target_frame


def test_stopgrad_loss_matches_base_loss_value() -> None:
    key = jax.random.key(0)
    base = init_jepa_base(key, embed_dim=32, hidden_dim=32)
    stopgrad = init_jepa_stopgrad(key, embed_dim=32, hidden_dim=32)
    batch = _batch()

    assert jnp.allclose(
        jepa_stopgrad_loss(stopgrad, batch), jepa_base_loss(base, batch)
    )


def test_ema_loss_is_finite() -> None:
    params = init_jepa_ema(jax.random.key(0), embed_dim=32, hidden_dim=32)
    loss = jepa_ema_loss(params, params.encoder, _batch())
    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_ema_target_receives_no_gradient() -> None:
    params = init_jepa_ema(jax.random.key(0), embed_dim=32, hidden_dim=32)
    batch = _batch()
    grads = jax.grad(
        lambda target_encoder: jepa_ema_loss(params, target_encoder, batch)
    )(params.encoder)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(grad == 0.0)


def test_sigreg_loss_is_finite_and_nonnegative() -> None:
    params = init_jepa_sigreg(jax.random.key(0), embed_dim=32, hidden_dim=32)
    loss = jepa_sigreg_loss(
        params, params.encoder, _batch(), key=jax.random.key(1), weight=0.1
    )
    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_sigreg_weight_zero_keeps_the_latent_loss() -> None:
    params = init_jepa_sigreg(jax.random.key(0), embed_dim=32, hidden_dim=32)
    batch = _batch()
    with_sigreg = jepa_sigreg_loss(
        params, params.encoder, batch, key=jax.random.key(1), weight=0.0
    )
    z_hat, _ = apply_jepa_sigreg(params, batch[0])
    z_target = apply_cnn(params.encoder, batch[1][..., None])
    latent = jnp.mean((z_hat - jax.lax.stop_gradient(z_target)) ** 2)
    assert jnp.allclose(with_sigreg, latent)


def test_ema_train_step_increments_optimizer_step() -> None:
    params = init_jepa_ema(jax.random.key(0), embed_dim=32, hidden_dim=32)
    state = EMAState(
        train_state=TrainState(
            params=params, opt_state=adamw_init(params)
        ),
        target_encoder=params.encoder,
    )

    next_state, loss = ema_train_step(
        state,
        _batch(),
        jepa_ema_loss,
        lambda p: p.encoder,
        lr=1e-3,
        weight_decay=0.0,
        decay=0.99,
    )

    assert int(next_state.train_state.opt_state.t) == 1
    assert jnp.isfinite(loss)
