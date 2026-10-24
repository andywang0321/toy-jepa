"""Fixed-capacity MLP readout trained on frozen features."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .. import constants
from .metrics import state_readout_metrics
from ..models.mlp import apply_mlp, init_mlp, mlp_mse_loss
from ..core.optim import adamw_init
from ..core.train import train_step
from ..core.types import ArrayLike, MLPParams, TrainState


def init_readout(
    key: jax.Array,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    state_dim: int = 8,
) -> MLPParams:
    """Initialize the ``embed -> hidden -> state`` readout."""
    return init_mlp(key, [embed_dim, hidden_dim, state_dim])


def apply_readout(params: MLPParams, features: jax.Array) -> jax.Array:
    """Map frozen features to normalized state predictions."""
    return apply_mlp(params, features)


def train_readout(
    key: jax.Array,
    train_features: ArrayLike,
    train_states: ArrayLike,
    val_features: ArrayLike,
    val_states: ArrayLike,
    *,
    num_steps: int = constants.READOUT_STEPS,
    batch_size: int = constants.READOUT_BATCH_SIZE,
    lr: float = constants.READOUT_LR,
    weight_decay: float = constants.READOUT_WEIGHT_DECAY,
    eval_every: int = constants.READOUT_EVAL_EVERY,
    patience: int = constants.READOUT_PATIENCE,
    min_delta: float = 1e-6,
    seed: int = 0,
) -> tuple[MLPParams, dict[str, float | int]]:
    """Train with AdamW and validation early stopping; return best params."""
    if num_steps < 0 or batch_size < 1 or eval_every < 1 or patience < 0:
        raise ValueError("invalid readout loop settings.")

    train_x = jnp.asarray(train_features, dtype=jnp.float32)
    train_y = jnp.asarray(train_states, dtype=jnp.float32)
    val_x = jnp.asarray(val_features, dtype=jnp.float32)
    val_y = jnp.asarray(val_states, dtype=jnp.float32)
    if train_x.shape[0] == 0 or val_x.shape[0] == 0:
        raise ValueError("readout training needs non-empty train and val sets.")
    if train_x.shape[0] != train_y.shape[0] or val_x.shape[0] != val_y.shape[0]:
        raise ValueError("features and states must have matching row counts.")

    params = init_readout(
        key,
        embed_dim=train_x.shape[-1],
        state_dim=train_y.shape[-1],
    )
    state = TrainState(params=params, opt_state=adamw_init(params))
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(
        state: TrainState[MLPParams],
        batch: tuple[jax.Array, jax.Array],
    ) -> tuple[TrainState[MLPParams], jax.Array]:
        return train_step(state, batch, mlp_mse_loss, lr=lr, weight_decay=weight_decay)

    @jax.jit
    def validation_loss(params: MLPParams, features: jax.Array, states: jax.Array) -> jax.Array:
        return jnp.mean((apply_readout(params, features) - states) ** 2)

    best_mse = float(validation_loss(state.params, val_x, val_y))
    best_params = state.params
    best_step = 0
    steps_run = 0
    stale_steps = 0

    for i in range(1, num_steps + 1):
        rows = rng.integers(0, train_x.shape[0], size=batch_size)
        state, _ = step(state, (train_x[rows], train_y[rows]))
        steps_run = i
        if i % eval_every == 0 or i == num_steps:
            value = float(validation_loss(state.params, val_x, val_y))
            if value < best_mse - min_delta:
                best_mse, best_params, best_step = value, state.params, i
                stale_steps = 0
            else:
                stale_steps += eval_every
            if stale_steps >= patience:
                break

    metrics: dict[str, float | int] = {
        "best_val_mse": best_mse,
        "best_step": best_step,
        "steps_run": steps_run,
        "num_train": int(train_x.shape[0]),
        "num_val": int(val_x.shape[0]),
    }
    return best_params, metrics


def fit_readout(
    key: jax.Array,
    train_features: ArrayLike,
    train_states: ArrayLike,
    val_features: ArrayLike,
    val_states: ArrayLike,
    test_features: ArrayLike,
    test_states: ArrayLike,
    *,
    seed: int,
) -> tuple[dict[str, float], dict[str, float | int]]:
    """Train the common readout and return test and validation metrics."""
    params, val_metrics = train_readout(
        key,
        train_features,
        train_states,
        val_features,
        val_states,
        seed=seed,
    )
    predictions = jax.jit(apply_readout)(params, jnp.asarray(test_features))
    test_metrics = state_readout_metrics(
        np.asarray(predictions),
        np.asarray(test_states),
        position_scale=constants.POSITION_SCALE,
        velocity_scale=constants.VELOCITY_SCALE,
    )
    return test_metrics, val_metrics
