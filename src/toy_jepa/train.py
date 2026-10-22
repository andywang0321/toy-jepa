"""Pure training functions, generic over the model and batch type."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

from .ema import ema_update
from .models.baseline import apply_baseline
from .models.cnn import apply_cnn
from .models.jepa_base import apply_jepa_base
from .models.mlp import apply_mlp
from .optim import adamw_update
from .sigreg import sigreg_loss
from .types import (
    BaselineParams,
    CNNParams,
    JEPABaseParams,
    JEPAEMAState,
    MLPParams,
    ParamsT,
    TrainState,
)

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


def jepa_stopgrad_loss(
    params: JEPABaseParams,
    batch: tuple[jax.Array, ...],
) -> jax.Array:
    """JEPA with a stop-gradient on the shared-encoder target."""
    context, target_frame = batch[0], batch[1]
    z_hat, _ = apply_jepa_base(params, context)
    z_target = apply_cnn(params.encoder, target_frame[..., None])
    return jnp.mean((z_hat - jax.lax.stop_gradient(z_target)) ** 2)


def jepa_ema_loss(
    params: JEPABaseParams,
    target_encoder: CNNParams,
    batch: tuple[jax.Array, ...],
) -> jax.Array:
    """JEPA with a stop-gradient target from the EMA encoder."""
    context, target_frame = batch[0], batch[1]
    z_hat, _ = apply_jepa_base(params, context)
    z_target = apply_cnn(target_encoder, target_frame[..., None])
    return jnp.mean((z_hat - jax.lax.stop_gradient(z_target)) ** 2)


def jepa_ema_sigreg_loss(
    params: JEPABaseParams,
    target_encoder: CNNParams,
    batch: tuple[jax.Array, ...],
    *,
    key: jax.Array,
    weight: float,
) -> jax.Array:
    """EMA JEPA loss plus SIGReg on the context feature ``c_t``.

    The online model runs once: ``c_t`` feeds both the predictor and SIGReg.
    """
    context, target_frame = batch[0], batch[1]
    z_hat, c_t = apply_jepa_base(params, context)
    z_target = apply_cnn(target_encoder, target_frame[..., None])
    latent_loss = jnp.mean(
        (z_hat - jax.lax.stop_gradient(z_target)) ** 2
    )
    return latent_loss + weight * sigreg_loss(c_t, key)


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


def ema_train_step(
    state: JEPAEMAState,
    batch: BatchT,
    loss_fn: Callable[[JEPABaseParams, CNNParams, BatchT], jax.Array],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    decay: float = 0.99,
) -> tuple[JEPAEMAState, jax.Array]:
    """Update the online model with AdamW, then EMA-update its target encoder.

    ``loss_fn`` receives ``(online_params, target_encoder, batch)`` and is
    differentiated only with respect to the online parameters.
    """
    loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
        state.train_state.params, state.target_encoder, batch
    )
    params, opt_state = adamw_update(
        state.train_state.params,
        grads,
        state.train_state.opt_state,
        lr=lr,
        weight_decay=weight_decay,
    )
    target_encoder = ema_update(
        state.target_encoder, params.encoder, decay
    )
    return (
        JEPAEMAState(
            train_state=TrainState(params=params, opt_state=opt_state),
            target_encoder=target_encoder,
        ),
        loss,
    )
