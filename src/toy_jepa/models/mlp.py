"""
An MLP in JAX
"""

import jax
import jax.numpy as jnp

from ..types import MLPParams


def init_mlp(key: jax.Array, sizes: list[int]) -> MLPParams:
    keys: jax.Array = jax.random.split(key, len(sizes) - 1)
    initializer = jax.nn.initializers.kaiming_uniform()
    params: MLPParams = []
    for layer_key, dim_in, dim_out in zip(keys, sizes[:-1], sizes[1:]):
        params.append({
            "w": initializer(layer_key, (dim_in, dim_out), jnp.float32),
            "b": jnp.zeros(dim_out, dtype=jnp.float32)
        })
    return params


def apply_mlp(params: MLPParams, x: jax.Array) -> jax.Array:
    for layer in params[:-1]:
        x = jax.nn.gelu(x @ layer["w"] + layer["b"])
    last = params[-1]
    return x @ last["w"] + last["b"]
