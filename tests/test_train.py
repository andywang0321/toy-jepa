"""Tests for the phase-2 Optax step."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from toy_jepa.mlp import init_mlp, mlp_mse_loss
from toy_jepa.train import TrainState, make_optimizer, train_step


def test_optax_step_reduces_loss() -> None:
    params = init_mlp(jax.random.key(0), [4, 8, 2])
    optimizer = make_optimizer(learning_rate=1e-2, weight_decay=0.0)
    state = TrainState(params=params, opt_state=optimizer.init(params))
    batch = (jnp.ones((8, 4)), jnp.zeros((8, 2)))

    before = float(mlp_mse_loss(state.params, batch))
    for _ in range(20):
        state, _ = train_step(state, batch, mlp_mse_loss, optimizer)
    after = float(mlp_mse_loss(state.params, batch))

    assert np.isfinite(after)
    assert after < before
