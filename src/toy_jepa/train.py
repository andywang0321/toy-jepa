"""Optax training step for phase-2 models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, NamedTuple, TypeVar, cast

import jax
import optax

ParamsT = TypeVar("ParamsT")
BatchT = TypeVar("BatchT")


class TrainState(NamedTuple, Generic[ParamsT]):
    params: ParamsT
    opt_state: optax.OptState


def make_optimizer(*, learning_rate: float, weight_decay: float) -> optax.GradientTransformation:
    """AdamW with the phase-2 defaults."""
    return optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def train_step(
    state: TrainState[ParamsT],
    batch: BatchT,
    loss_fn: Callable[[ParamsT, BatchT], jax.Array],
    optimizer: optax.GradientTransformation,
) -> tuple[TrainState[ParamsT], jax.Array]:
    """Apply one Optax update to ``loss_fn``."""
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
    updates, opt_state = optimizer.update(grads, state.opt_state, cast(Any, state.params))
    params = cast(ParamsT, optax.apply_updates(cast(Any, state.params), updates))
    return TrainState(params=params, opt_state=opt_state), cast(jax.Array, loss)


def train_step_with_key(
    state: TrainState[ParamsT],
    batch: BatchT,
    key: jax.Array,
    loss_fn: Callable[[ParamsT, BatchT, jax.Array], jax.Array],
    optimizer: optax.GradientTransformation,
) -> tuple[TrainState[ParamsT], jax.Array]:
    """Apply one Optax update to a keyed loss."""
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch, key)
    updates, opt_state = optimizer.update(
        grads, state.opt_state, cast(Any, state.params)
    )
    params = cast(ParamsT, optax.apply_updates(cast(Any, state.params), updates))
    return TrainState(params=params, opt_state=opt_state), cast(jax.Array, loss)
