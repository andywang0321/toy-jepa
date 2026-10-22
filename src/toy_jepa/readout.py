"""Fixed-capacity MLP readout on frozen features.

The readout is the common external metric for every arm: freeze the model,
extract feature arrays, then train ``128 -> 256 -> 8`` (GELU hidden, linear
output) to predict the normalized state at ``t + k``. It never backpropagates
into the encoder because its inputs are plain arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .constants import POSITION_SCALE, VELOCITY_SCALE
from .metrics import state_readout_metrics
from .models.mlp import apply_mlp, init_mlp
from .optim import adamw_init
from .train import mlp_mse_loss, train_step
from .types import Float32Array, MLPParams, TrainState


def init_readout(
    key: jax.Array,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    state_dim: int = 8,
) -> MLPParams:
    """Initialize the fixed-capacity readout: embed -> hidden -> state."""
    return init_mlp(key, [embed_dim, hidden_dim, state_dim])


def apply_readout(params: MLPParams, features: jax.Array) -> jax.Array:
    """Map frozen features to normalized state predictions."""
    return apply_mlp(params, features)


def train_readout(
    key: jax.Array,
    train_features: Float32Array,
    train_states: Float32Array,
    val_features: Float32Array,
    val_states: Float32Array,
    *,
    num_steps: int = 2_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    eval_every: int = 50,
    patience: int = 500,
    min_delta: float = 1e-6,
    seed: int = 0,
    position_scale: float = POSITION_SCALE,
    velocity_scale: float = VELOCITY_SCALE,
) -> tuple[MLPParams, dict[str, float | int]]:
    """Train the readout on frozen features with validation early stopping.

    Args:
        key: PRNG key for readout initialization.
        train_features: ``(N, embed_dim)`` frozen training features.
        train_states: ``(N, 8)`` normalized target states.
        val_features: ``(M, embed_dim)`` frozen validation features.
        val_states: ``(M, 8)`` normalized validation targets.
        num_steps: maximum AdamW steps.
        batch_size: minibatch size (with replacement).
        lr: AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
        eval_every: validation/evaluation interval in steps.
        patience: stop after this many steps without a validation improvement.
        min_delta: minimum decrease in validation MSE to count as an improvement.
        seed: NumPy RNG seed for minibatch sampling.
        position_scale: pixels per normalized position unit.
        velocity_scale: pixels/frame per normalized velocity unit.

    Returns:
        ``(best_params, metrics)`` where metrics are the validation state
        metrics from the best step, plus the step counts.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    if batch_size < 1 or eval_every < 1 or patience < 0:
        raise ValueError("batch_size and eval_every must be positive.")

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
        return train_step(
            state,
            batch,
            mlp_mse_loss,
            lr=lr,
            weight_decay=weight_decay,
        )

    @jax.jit
    def validation_loss(
        params: MLPParams, features: jax.Array, states: jax.Array
    ) -> jax.Array:
        return jnp.mean((apply_readout(params, features) - states) ** 2)

    best_value = float(validation_loss(state.params, val_x, val_y))
    best_params = state.params
    best_step = 0
    steps_run = 0
    stale_steps = 0
    num_train = train_x.shape[0]

    for i in range(1, num_steps + 1):
        rows = rng.integers(0, num_train, size=batch_size)
        state, _ = step(state, (train_x[rows], train_y[rows]))
        steps_run = i

        if i % eval_every == 0 or i == num_steps:
            value = float(validation_loss(state.params, val_x, val_y))
            if value < best_value - min_delta:
                best_value = value
                best_params = state.params
                best_step = i
                stale_steps = 0
            else:
                stale_steps += eval_every
            if stale_steps >= patience:
                break

    predictions = jax.jit(apply_readout)(best_params, val_x)
    metrics = state_readout_metrics(
        np.asarray(predictions),
        np.asarray(val_y),
        position_scale=position_scale,
        velocity_scale=velocity_scale,
    )
    metrics.update(
        {
            "best_val_mse": best_value,
            "best_step": best_step,
            "steps_run": steps_run,
            "num_train": int(num_train),
            "num_val": int(val_x.shape[0]),
        }
    )
    return best_params, metrics
