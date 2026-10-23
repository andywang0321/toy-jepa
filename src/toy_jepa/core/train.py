"""Pure, model-agnostic training steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax

from ..models.ema import ema_update
from .optim import adamw_update
from .types import CNNParams, EMAState, ParamsT, TrainState

BatchT = TypeVar("BatchT")


def train_step(
    state: TrainState[ParamsT],
    batch: BatchT,
    loss_fn: Callable[[ParamsT, BatchT], jax.Array],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> tuple[TrainState[ParamsT], jax.Array]:
    """Differentiate ``loss_fn`` and apply one AdamW update."""
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
    params, opt_state = adamw_update(
        state.params, grads, state.opt_state, lr=lr, weight_decay=weight_decay
    )
    return TrainState(params=params, opt_state=opt_state), loss


def ema_train_step(
    state: EMAState[ParamsT],
    batch: BatchT,
    loss_fn: Callable[[ParamsT, CNNParams, BatchT], jax.Array],
    get_encoder: Callable[[ParamsT], CNNParams],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    decay: float = 0.99,
) -> tuple[EMAState[ParamsT], jax.Array]:
    """Update the online model, then move its target encoder toward it."""
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
    target_encoder = ema_update(state.target_encoder, get_encoder(params), decay)
    return (
        EMAState(
            train_state=TrainState(params=params, opt_state=opt_state),
            target_encoder=target_encoder,
        ),
        loss,
    )
