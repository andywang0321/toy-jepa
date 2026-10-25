"""A small temporal transformer over frame tokens."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core.types import AffineParams, MLPParams, NormParams
from ..models.mlp import apply_mlp, init_mlp
from ..models.norm import apply_layer_norm, init_layer_norm


class AttentionParams(NamedTuple):
    query: AffineParams
    key: AffineParams
    value: AffineParams
    out: AffineParams


class BlockParams(NamedTuple):
    ln1: NormParams
    attention: AttentionParams
    ln2: NormParams
    mlp: MLPParams


class TemporalParams(NamedTuple):
    blocks: list[BlockParams]
    ln: NormParams


def _init_affine(key: jax.Array, dim: int) -> AffineParams:
    return init_mlp(key, [dim, dim]).blocks[0]


def init_attention(key: jax.Array, dim: int) -> AttentionParams:
    keys = jax.random.split(key, 4)
    return AttentionParams(
        query=_init_affine(keys[0], dim),
        key=_init_affine(keys[1], dim),
        value=_init_affine(keys[2], dim),
        out=_init_affine(keys[3], dim),
    )


def init_temporal(
    key: jax.Array,
    *,
    dim: int = 128,
    heads: int = 4,
    layers: int = 2,
    mlp_ratio: int = 4,
) -> TemporalParams:
    keys = jax.random.split(key, layers + 1)
    blocks = [
        BlockParams(
            ln1=init_layer_norm(dim),
            attention=init_attention(keys[layer], dim),
            ln2=init_layer_norm(dim),
            mlp=init_mlp(
                jax.random.fold_in(keys[layer], 1),
                [dim, mlp_ratio * dim, dim],
            ),
        )
        for layer in range(layers)
    ]
    return TemporalParams(blocks=blocks, ln=init_layer_norm(dim))


def _split_heads(x: jax.Array, heads: int) -> jax.Array:
    batch, length, dim = x.shape
    return x.reshape(batch, length, heads, dim // heads).transpose(0, 2, 1, 3)


def _merge_heads(x: jax.Array) -> jax.Array:
    batch, heads, length, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, heads * head_dim)


def apply_attention(
    params: AttentionParams,
    query: jax.Array,
    context: jax.Array,
    mask: jax.Array,
    *,
    heads: int,
) -> jax.Array:
    """Multi-head attention with a boolean key mask."""
    q = _split_heads(query @ params.query.w + params.query.b, heads)
    k = _split_heads(context @ params.key.w + params.key.b, heads)
    v = _split_heads(context @ params.value.w + params.value.b, heads)

    scale = jnp.sqrt(q.shape[-1])
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / scale
    scores = jnp.where(mask[:, None, None, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    return _merge_heads(attended) @ params.out.w + params.out.b


def _positional(ages: jax.Array, dim: int) -> jax.Array:
    frequencies = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    angles = ages[..., None] * 10000.0 ** (-frequencies)
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def apply_temporal(
    params: TemporalParams,
    tokens: jax.Array,
    mask: jax.Array,
    *,
    heads: int,
) -> jax.Array:
    """Encode a padded frame-token sequence into one context vector."""
    length = tokens.shape[1]
    ages = mask.sum(axis=1, keepdims=True) - 1 - jnp.arange(length)[None, :]
    ages = jnp.where(mask, ages, 0)
    x = tokens + _positional(ages, tokens.shape[-1]) * mask[..., None]

    for block in params.blocks:
        h = apply_layer_norm(block.ln1, x)
        x = x + apply_attention(block.attention, h, h, mask, heads=heads)
        h = apply_layer_norm(block.ln2, x)
        x = x + apply_mlp(block.mlp, h)

    weights = mask[..., None].astype(jnp.float32)
    pooled = jnp.sum(x * weights, axis=1) / jnp.sum(weights, axis=1)
    return apply_layer_norm(params.ln, pooled)
