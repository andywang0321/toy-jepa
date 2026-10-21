"""Tests for the MLP model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.models.mlp import apply_mlp, init_mlp


def test_init_mlp_shapes_and_dtypes() -> None:
    params = init_mlp(jax.random.key(0), [4, 8, 2])

    assert len(params.blocks) == 2
    assert params.blocks[0].w.shape == (4, 8)
    assert params.blocks[0].b.shape == (8,)
    assert params.blocks[1].w.shape == (8, 2)
    assert params.blocks[1].b.shape == (2,)

    for layer in params.blocks:
        assert layer.w.dtype == jnp.float32
        assert layer.b.dtype == jnp.float32
        assert jnp.all(layer.b == 0)
        assert jnp.all(jnp.isfinite(layer.w))


def test_init_mlp_uses_independent_keys() -> None:
    params = init_mlp(jax.random.key(0), [8, 8, 8])

    # Same weight shape, different layers: split keys should not collide.
    assert not jnp.array_equal(params.blocks[0].w, params.blocks[1].w)


def test_apply_mlp_final_layer_is_linear() -> None:
    params = init_mlp(jax.random.key(0), [4, 2])
    x = jnp.ones((3, 4))

    expected = x @ params.blocks[0].w + params.blocks[0].b
    assert jnp.allclose(apply_mlp(params, x), expected)


def test_apply_mlp_matches_manual_two_layer_forward() -> None:
    params = init_mlp(jax.random.key(0), [4, 8, 2])
    x = jnp.ones((3, 4))

    hidden = jax.nn.gelu(x @ params.blocks[0].w + params.blocks[0].b)
    expected = hidden @ params.blocks[1].w + params.blocks[1].b

    output = apply_mlp(params, x)
    assert output.shape == (3, 2)
    assert jnp.allclose(output, expected)


def test_apply_mlp_is_jittable_and_vmappable() -> None:
    params = init_mlp(jax.random.key(0), [4, 8, 2])
    x = jnp.ones((5, 4))

    batched = jax.jit(apply_mlp)(params, x)
    per_sample = jax.vmap(lambda row: apply_mlp(params, row))(x)

    assert batched.shape == (5, 2)
    assert jnp.allclose(batched, per_sample)


def test_apply_mlp_has_finite_gradients() -> None:
    params = init_mlp(jax.random.key(0), [4, 8, 2])
    x = jnp.ones((5, 4))

    def loss(p):
        return jnp.mean(apply_mlp(p, x) ** 2)

    grads = jax.grad(loss)(params)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))
