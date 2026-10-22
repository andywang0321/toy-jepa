"""Tests for the EMA target encoder and the EMA train step."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.ema import ema_update
from toy_jepa.models.cnn import init_cnn
from toy_jepa.models.jepa_base import init_jepa_base
from toy_jepa.optim import adamw_init
from toy_jepa.train import ema_train_step, jepa_ema_loss
from toy_jepa.types import CNNParams, JEPABaseParams, JEPAEMAState, TrainState


def _tree_distance(left: object, right: object) -> float:
    """Squared Euclidean distance between two parameter trees."""
    return float(
        sum(
            jnp.sum((a - b) ** 2)
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
        )
    )


def test_ema_update_moves_target_toward_online() -> None:
    target = init_cnn(jax.random.key(0), embed_dim=16)
    online = jax.tree.map(lambda leaf: leaf + 1.0, target)

    updated = ema_update(target, online, 0.9)

    assert _tree_distance(updated, online) < _tree_distance(target, online)
    for before, after, reference in zip(
        jax.tree.leaves(target),
        jax.tree.leaves(updated),
        jax.tree.leaves(online),
    ):
        assert jnp.allclose(after, 0.9 * before + 0.1 * reference)


def test_ema_train_step_updates_online_and_target() -> None:
    params = init_jepa_base(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )
    state = JEPAEMAState(
        train_state=TrainState(
            params=params, opt_state=adamw_init(params)
        ),
        target_encoder=params.encoder,
    )
    context = jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)
    target_frame = jnp.ones((2, 64, 64), dtype=jnp.float32)
    target_state = jnp.zeros((2, 8), dtype=jnp.float32)

    next_state, loss = ema_train_step(
        state,
        (context, target_frame, target_state),
        jepa_ema_loss,
        lr=1e-3,
        weight_decay=0.0,
        decay=0.5,
    )

    assert int(next_state.train_state.opt_state.t) == 1
    assert jnp.isfinite(loss)
    assert isinstance(next_state.train_state.params, JEPABaseParams)
    assert isinstance(next_state.target_encoder, CNNParams)
    assert jax.tree.structure(next_state.train_state.params) == (
        jax.tree.structure(params)
    )
    assert jax.tree.structure(next_state.target_encoder) == (
        jax.tree.structure(params.encoder)
    )
    assert _tree_distance(
        next_state.train_state.params, state.train_state.params
    ) > 0.0
    assert _tree_distance(
        next_state.target_encoder, state.target_encoder
    ) > 0.0
