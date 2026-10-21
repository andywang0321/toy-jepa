"""
Pure training functions
"""

import jax
import jax.numpy as jnp

from .types import MLPParams, TrainState
from .models.mlp import apply_mlp
from .optim import adamw_update

def mse_loss(params: MLPParams, x: jax.Array, y: jax.Array) -> jax.Array:
    y_hat: jax.Array = apply_mlp(params, x)
    return jnp.mean((y - y_hat)**2)

def train_step(
    state: TrainState,
    batch: tuple[jax.Array, jax.Array], # (Xs, ys)
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.
) -> tuple[TrainState, jax.Array]: # (TrainState, loss)
    x, y = batch
    loss, grads = jax.value_and_grad(mse_loss)(state.params, x, y)
    params, opt_state = adamw_update(state.params, grads, state.opt_state, lr=lr, weight_decay=weight_decay)
    return TrainState(params, opt_state), loss
