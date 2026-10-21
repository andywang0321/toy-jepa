"""Pure training functions, generic over the model and batch type."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

from .models.mlp import apply_mlp
from .models.pixel import apply_pixel_model
from .optim import adamw_update
from .types import MLPParams, ParamsT, PixelParams, TrainState

BatchT = TypeVar("BatchT")


def mlp_mse_loss(
    params: MLPParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Mean squared error of the MLP prediction."""
    x, y = batch
    y_hat = apply_mlp(params, x)
    return jnp.mean((y - y_hat) ** 2)


def pixel_mse_loss(
    params: PixelParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Mean squared error of the predicted future frame."""
    context, target = batch
    pixels, _, _ = apply_pixel_model(params, context)
    return jnp.mean((pixels - target) ** 2)


def train_step(
    state: TrainState[ParamsT],
    batch: BatchT,
    loss_fn: Callable[[ParamsT, BatchT], jax.Array],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> tuple[TrainState[ParamsT], jax.Array]:
    """One generic training step: differentiate ``loss_fn`` and apply AdamW."""
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
    params, opt_state = adamw_update(
        state.params, grads, state.opt_state, lr=lr, weight_decay=weight_decay
    )
    return TrainState(params=params, opt_state=opt_state), loss
