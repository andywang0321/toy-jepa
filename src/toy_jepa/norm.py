"""Functional normalization primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import NormParams

GROUPS = 8


def init_group_norm(channels: int) -> NormParams:
    """Initialize the affine parameters of a group norm."""
    return NormParams(
        gamma=jnp.ones(channels, dtype=jnp.float32),
        beta=jnp.zeros(channels, dtype=jnp.float32),
    )


def init_layer_norm(dim: int) -> NormParams:
    """Initialize the affine parameters of a layer norm."""
    return NormParams(
        gamma=jnp.ones(dim, dtype=jnp.float32),
        beta=jnp.zeros(dim, dtype=jnp.float32),
    )


def apply_group_norm(
    params: NormParams,
    x: jax.Array,
    *,
    groups: int = GROUPS,
    eps: float = 1e-5,
) -> jax.Array:
    """Apply group norm over the channel dimension of an NHWC tensor."""
    batch, height, width, channels = x.shape
    grouped = x.reshape(batch, height, width, groups, channels // groups)

    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    var = jnp.var(grouped, axis=(1, 2, 4), keepdims=True)
    normalized = (grouped - mean) * jax.lax.rsqrt(var + eps)
    normalized = normalized.reshape(batch, height, width, channels)

    return normalized * params.gamma + params.beta


def apply_layer_norm(
    params: NormParams,
    x: jax.Array,
    *,
    eps: float = 1e-5,
) -> jax.Array:
    """Apply layer norm over the last dimension."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * params.gamma + params.beta
