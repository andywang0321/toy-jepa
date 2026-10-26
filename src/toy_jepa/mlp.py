"""A minimal functional MLP."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import AffineParams, MLPParams


def init_mlp(key: jax.Array, sizes: list[int]) -> MLPParams:
    """Initialize a list of affine layers with He-uniform weights."""
    keys = jax.random.split(key, len(sizes) - 1)
    initializer = jax.nn.initializers.kaiming_uniform()
    blocks = [
        AffineParams(
            w=initializer(layer_key, (dim_in, dim_out), jnp.float32),
            b=jnp.zeros(dim_out, dtype=jnp.float32),
        )
        for layer_key, dim_in, dim_out in zip(keys, sizes[:-1], sizes[1:])
    ]
    return MLPParams(blocks=blocks)


def apply_mlp(params: MLPParams, x: jax.Array) -> jax.Array:
    """Apply all hidden layers with GELU and the final layer linearly."""
    for layer in params.blocks[:-1]:
        x = jax.nn.gelu(x @ layer.w + layer.b)
    last = params.blocks[-1]
    return x @ last.w + last.b


def mlp_mse_loss(
    params: MLPParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Mean squared error of the MLP prediction."""
    x, y = batch
    return jnp.mean((apply_mlp(params, x) - y) ** 2)
