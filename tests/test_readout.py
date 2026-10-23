"""Tests for the fixed-capacity state readout."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from toy_jepa.models.mlp import mlp_mse_loss
from toy_jepa.readout import apply_readout, init_readout, train_readout


def _readout_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random frozen features and independent random state targets."""
    rng = np.random.default_rng(0)
    train_x = rng.normal(size=(256, 16)).astype(np.float32)
    train_y = rng.normal(size=(256, 8)).astype(np.float32)
    val_x = rng.normal(size=(128, 16)).astype(np.float32)
    val_y = rng.normal(size=(128, 8)).astype(np.float32)
    return train_x, train_y, val_x, val_y


def test_readout_shapes() -> None:
    params = init_readout(
        jax.random.key(0), embed_dim=16, hidden_dim=32, state_dim=8
    )
    features = jnp.ones((5, 16), dtype=jnp.float32)

    predictions = apply_readout(params, features)

    assert predictions.shape == (5, 8)
    assert len(params.blocks) == 2
    assert params.blocks[0].w.shape == (16, 32)
    assert params.blocks[1].w.shape == (32, 8)
    assert jnp.all(jnp.isfinite(predictions))


def test_readout_has_finite_gradients() -> None:
    params = init_readout(
        jax.random.key(0), embed_dim=16, hidden_dim=32, state_dim=8
    )
    features = jnp.ones((5, 16), dtype=jnp.float32)
    states = jnp.zeros((5, 8), dtype=jnp.float32)

    grads = jax.grad(lambda p: mlp_mse_loss(p, (features, states)))(params)

    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))


def test_readout_early_stops() -> None:
    train_x, train_y, val_x, val_y = _readout_data()

    _, metrics = train_readout(
        jax.random.key(0),
        train_x,
        train_y,
        val_x,
        val_y,
        num_steps=500,
        batch_size=64,
        eval_every=25,
        patience=0,
        seed=0,
    )

    assert metrics["steps_run"] == 25
    assert metrics["best_step"] <= metrics["steps_run"]
    assert metrics["num_train"] == 256
    assert metrics["num_val"] == 128
    assert np.isfinite(metrics["best_val_mse"])


def test_readout_learns_a_linear_state_map() -> None:
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(16, 8)).astype(np.float32) / 4.0
    train_x = rng.normal(size=(512, 16)).astype(np.float32)
    val_x = rng.normal(size=(256, 16)).astype(np.float32)
    train_y = (train_x @ weights).astype(np.float32)
    val_y = (val_x @ weights).astype(np.float32)
    baseline_mse = float(np.mean(val_y**2))

    params, metrics = train_readout(
        jax.random.key(0),
        train_x,
        train_y,
        val_x,
        val_y,
        num_steps=400,
        batch_size=128,
        lr=1e-2,
        eval_every=20,
        patience=100,
        seed=1,
    )

    assert metrics["best_val_mse"] < 0.2 * baseline_mse
    predictions = jax.jit(apply_readout)(params, jnp.asarray(val_x))
    assert predictions.shape == (256, 8)
    assert jnp.all(jnp.isfinite(predictions))
