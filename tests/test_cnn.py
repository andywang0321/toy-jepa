"""Tests for the convolutional frame encoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.cnn import apply_cnn, init_cnn


def test_cnn_output_shape() -> None:
    params = init_cnn(jax.random.key(0))
    x = jnp.ones((2, 64, 64, 1), dtype=jnp.float32)

    z = apply_cnn(params, x)

    assert z.shape == (2, 128)
    assert z.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(z))


def test_cnn_is_jittable() -> None:
    params = init_cnn(jax.random.key(0))
    x = jnp.ones((2, 64, 64, 1), dtype=jnp.float32)

    eager = apply_cnn(params, x)
    compiled = jax.jit(apply_cnn)(params, x)

    assert jnp.allclose(eager, compiled, rtol=1e-3, atol=1e-4)


def test_cnn_has_finite_gradients() -> None:
    params = init_cnn(jax.random.key(0))
    x = jnp.ones((2, 64, 64, 1), dtype=jnp.float32)

    def loss(p):
        return jnp.mean(apply_cnn(p, x) ** 2)

    grads = jax.grad(loss)(params)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))


def test_cnn_depends_on_input() -> None:
    params = init_cnn(jax.random.key(0))
    zeros = jnp.zeros((1, 64, 64, 1), dtype=jnp.float32)
    ones = jnp.ones((1, 64, 64, 1), dtype=jnp.float32)

    assert not jnp.allclose(
        apply_cnn(params, zeros),
        apply_cnn(params, ones),
    )
