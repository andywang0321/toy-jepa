"""Tests for the SIGReg regularizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.sigreg import sigreg_loss


def test_sigreg_loss_is_finite_and_nonnegative() -> None:
    features = jax.random.normal(jax.random.key(0), (64, 16))

    loss = sigreg_loss(features, jax.random.key(1))

    assert jnp.isfinite(loss)
    assert loss >= 0.0


def test_sigreg_penalizes_constant_features() -> None:
    # A nonzero constant feature cloud has a characteristic function that
    # visibly misses the standard-normal target.
    features = jnp.full((64, 8), 2.0, dtype=jnp.float32)

    loss = sigreg_loss(features, jax.random.key(0))

    assert jnp.isfinite(loss)
    assert loss > 0.01


def test_sigreg_standard_normal_features_are_not_pathological() -> None:
    features = jax.random.normal(jax.random.key(0), (512, 16))

    loss = sigreg_loss(features, jax.random.key(1))

    assert jnp.isfinite(loss)
    assert loss < 0.1
