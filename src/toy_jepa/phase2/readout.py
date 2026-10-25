"""Frozen-feature probes for phase 2."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..core.types import MLPParams
from ..evaluation.metrics import state_readout_metrics
from ..models.mlp import apply_mlp, init_mlp


class ProbeResult(NamedTuple):
    params: MLPParams
    val_loss: float
    steps_run: int


def _fit(
    key: jax.Array,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    loss_fn: Callable[[MLPParams, tuple[jax.Array, jax.Array]], jax.Array],
    *,
    output_dim: int,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ProbeResult:
    train_x_jax = jnp.asarray(train_x, dtype=jnp.float32)
    train_y_jax = jnp.asarray(train_y, dtype=jnp.float32)
    val_x_jax = jnp.asarray(val_x, dtype=jnp.float32)
    val_y_jax = jnp.asarray(val_y, dtype=jnp.float32)
    params = init_mlp(key, [train_x_jax.shape[-1], 256, output_dim])
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(
        params: MLPParams,
        opt_state: optax.OptState,
        batch: tuple[jax.Array, jax.Array],
    ) -> tuple[MLPParams, Any, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = cast(MLPParams, optax.apply_updates(params, updates))
        return new_params, opt_state, cast(jax.Array, loss)

    @jax.jit
    def validation(params: MLPParams) -> jax.Array:
        return loss_fn(params, (val_x_jax, val_y_jax))

    best_params = params
    best_loss = float(validation(params))
    for step_number in range(1, num_steps + 1):
        rows = rng.integers(0, train_x_jax.shape[0], size=batch_size)
        params, opt_state, _ = step(params, opt_state, (train_x_jax[rows], train_y_jax[rows]))
        if step_number % 100 == 0 or step_number == num_steps:
            value = float(validation(params))
            if value < best_loss:
                best_loss, best_params = value, params
    return ProbeResult(params=best_params, val_loss=best_loss, steps_run=num_steps)


def fit_state_probe(
    key: jax.Array,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ProbeResult:
    def loss(params: MLPParams, batch: tuple[jax.Array, jax.Array]) -> jax.Array:
        x, y = batch
        return jnp.mean((apply_mlp(params, x) - y) ** 2)

    return _fit(
        key,
        train_x,
        train_y,
        val_x,
        val_y,
        loss,
        output_dim=8,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )


def fit_digit_probe(
    key: jax.Array,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ProbeResult:
    def loss(params: MLPParams, batch: tuple[jax.Array, jax.Array]) -> jax.Array:
        x, y = batch
        labels = y.astype(jnp.int32)
        logits = apply_mlp(params, x).reshape(-1, 2, 10)
        forward = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, 0], labels[:, 0]
        ) + optax.softmax_cross_entropy_with_integer_labels(logits[:, 1], labels[:, 1])
        reverse = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, 0], labels[:, 1]
        ) + optax.softmax_cross_entropy_with_integer_labels(logits[:, 1], labels[:, 0])
        return jnp.mean(jnp.minimum(forward, reverse))

    return _fit(
        key,
        train_x,
        train_y,
        val_x,
        val_y,
        loss,
        output_dim=20,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )


def fit_bounce_probe(
    key: jax.Array,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ProbeResult:
    def loss(params: MLPParams, batch: tuple[jax.Array, jax.Array]) -> jax.Array:
        x, y = batch
        logits = apply_mlp(params, x)[:, 0]
        return optax.sigmoid_binary_cross_entropy(logits, y).mean()

    return _fit(
        key,
        train_x,
        train_y,
        val_x,
        val_y,
        loss,
        output_dim=1,
        num_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )


def state_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return state_readout_metrics(predictions, targets)


def digit_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    predictions = logits.reshape(-1, 2, 10).argmax(axis=-1)
    forward = (predictions[:, 0] == labels[:, 0]) & (predictions[:, 1] == labels[:, 1])
    reverse = (predictions[:, 0] == labels[:, 1]) & (predictions[:, 1] == labels[:, 0])
    return float(np.mean(forward | reverse))


def bounce_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predictions = logits[:, 0] > 0.0
    positives = labels > 0.5
    accuracy = float(np.mean(predictions == positives))
    positive_rate = float(np.mean(positives))
    majority = max(positive_rate, 1.0 - positive_rate)
    tpr = float(np.mean(predictions[positives])) if np.any(positives) else float("nan")
    tnr = float(np.mean(~predictions[~positives])) if np.any(~positives) else float("nan")
    return {
        "accuracy": accuracy,
        "positive_rate": positive_rate,
        "majority": majority,
        "balanced_accuracy": 0.5 * (tpr + tnr),
    }
