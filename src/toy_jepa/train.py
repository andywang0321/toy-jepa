"""Pure training functions, generic over the model and batch type."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

from .models.baseline import apply_baseline
from .models.cnn import apply_cnn
from .models.jepa_base import apply_jepa_base
from .models.mlp import apply_mlp
from .optim import adamw_update
from .types import BaselineParams, JEPABaseParams, MLPParams, ParamsT, TrainState

BatchT = TypeVar("BatchT")


def mlp_mse_loss(
    params: MLPParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Mean squared error of the MLP prediction."""
    x, y = batch
    y_hat = apply_mlp(params, x)
    return jnp.mean((y - y_hat) ** 2)


def baseline_pixel_loss(
    params: BaselineParams,
    batch: tuple[jax.Array, ...],
) -> jax.Array:
    """Pixel MSE for the pixel-only baseline.

    ``batch`` is ``(context, target_pixels, ...)``; extra entries such as the
    target state used by the post-hoc readout are ignored.
    """
    context, target_pixels = batch[0], batch[1]
    pixels, _, _ = apply_baseline(params, context)
    return jnp.mean((pixels - target_pixels) ** 2)


def jepa_base_loss(
    params: JEPABaseParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Barebones JEPA: MSE between predicted and shared-encoder targets."""
    context, target_frame = batch
    z_hat, _ = apply_jepa_base(params, context)
    z_target = apply_cnn(params.encoder, target_frame[..., None])
    return jnp.mean((z_hat - z_target) ** 2)


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
