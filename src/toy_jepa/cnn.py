"""Convolution blocks and the CNN built from them.

``init_conv``/``apply_conv`` are the primitive. ``init_conv_block`` adds a
normalization and GELU. ``init_cnn`` stacks blocks, flattens the spatial map,
projects to an embedding, and applies a final layer norm.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import AffineParams, CNNParams, ConvBlockParams, MLPParams
from .mlp import apply_mlp, init_mlp
from .norm import (
    GROUPS,
    apply_group_norm,
    apply_layer_norm,
    init_group_norm,
    init_layer_norm,
)


def init_conv(key: jax.Array, in_channels: int, out_channels: int, kernel_size: int = 3) -> AffineParams:
    """Initialize a ``(kernel, kernel, in, out)`` convolution."""
    initializer = jax.nn.initializers.kaiming_uniform()
    weight = initializer(key, (kernel_size, kernel_size, in_channels, out_channels), jnp.float32)
    return AffineParams(w=weight, b=jnp.zeros(out_channels, dtype=jnp.float32))


def apply_conv(params: AffineParams, x: jax.Array, *, stride: int = 2) -> jax.Array:
    """Apply an NHWC convolution with ``SAME`` padding."""
    y = jax.lax.conv_general_dilated(
        x,
        params.w,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + params.b


def init_conv_block(key: jax.Array, in_channels: int, out_channels: int) -> ConvBlockParams:
    """Initialize one ``conv -> group norm`` block."""
    return ConvBlockParams(
        conv=init_conv(key, in_channels, out_channels),
        norm=init_group_norm(out_channels),
    )


def apply_conv_block(params: ConvBlockParams, x: jax.Array) -> jax.Array:
    """Apply ``conv -> group norm -> GELU``."""
    x = apply_conv(params.conv, x)
    x = apply_group_norm(params.norm, x)
    return jax.nn.gelu(x)


def init_cnn(
    key: jax.Array,
    *,
    in_channels: int = 1,
    input_size: int = 64,
    channels: tuple[int, ...] = (32, 64, 128),
    embed_dim: int = 128,
) -> CNNParams:
    """Initialize a CNN made of ``len(channels)`` conv blocks."""
    if len(channels) == 0:
        raise ValueError("channels must contain at least one width.")
    if any(width % GROUPS != 0 for width in channels):
        raise ValueError(f"every channel count must be divisible by {GROUPS}.")

    spatial = input_size // (2 ** len(channels))
    if spatial <= 0:
        raise ValueError("input_size is too small for the number of conv layers.")
    flatten_dim = spatial * spatial * channels[-1]

    keys = jax.random.split(key, len(channels) + 1)
    blocks: list[ConvBlockParams] = []
    current_channels = in_channels

    for layer_key, out_channels in zip(keys[:-1], channels):
        blocks.append(init_conv_block(layer_key, current_channels, out_channels))
        current_channels = out_channels

    fc: MLPParams = init_mlp(keys[-1], [flatten_dim, embed_dim])
    return CNNParams(blocks=blocks, fc=fc, ln=init_layer_norm(embed_dim))


def apply_cnn(params: CNNParams, x: jax.Array) -> jax.Array:
    """Encode one frame ``(B, H, W, C)`` into ``(B, embed_dim)``."""
    for block in params.blocks:
        x = apply_conv_block(block, x)

    x = x.reshape(x.shape[0], -1)
    x = apply_mlp(params.fc, x)
    return apply_layer_norm(params.ln, x)
