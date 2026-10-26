"""Small conditional convolutional velocity field."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cnn import apply_conv, init_conv, init_conv_block
from .mlp import MLPParams, apply_mlp, init_mlp
from .norm import apply_group_norm
from .types import AffineParams, ConvBlockParams


class FlowCNNParams(NamedTuple):
    encoder: list[ConvBlockParams]
    cond: MLPParams
    film: MLPParams
    decoder: list[ConvBlockParams]
    out: AffineParams


def _block(params: ConvBlockParams, x: jax.Array, *, stride: int) -> jax.Array:
    x = apply_conv(params.conv, x, stride=stride)
    x = apply_group_norm(params.norm, x)
    return jax.nn.gelu(x)


def _up(x: jax.Array) -> jax.Array:
    batch, height, width, channels = x.shape
    return jax.image.resize(
        x, (batch, 2 * height, 2 * width, channels), method="nearest"
    )


def init_flow_cnn(
    key: jax.Array,
    *,
    cond_dim: int,
    widths: tuple[int, int, int] = (32, 64, 128),
) -> FlowCNNParams:
    keys = jax.random.split(key, 9)
    encoder = [
        init_conv_block(keys[index], in_ch, out_ch)
        for index, (in_ch, out_ch) in enumerate(zip((1, *widths[:-1]), widths))
    ]
    decoder = [
        init_conv_block(keys[3 + index], in_ch, out_ch)
        for index, (in_ch, out_ch) in enumerate(zip(tuple(reversed(widths)), (64, 32, 16)))
    ]
    return FlowCNNParams(
        encoder=encoder,
        cond=init_mlp(keys[7], [cond_dim, 256]),
        film=init_mlp(keys[8], [256, 2 * widths[-1]]),
        decoder=decoder,
        out=init_conv(jax.random.fold_in(key, 9), 16, 1),
    )


def apply_flow_cnn(
    params: FlowCNNParams,
    x: jax.Array,
    cond: jax.Array,
) -> jax.Array:
    """Map a noisy frame and conditioning vector to a velocity field."""
    h = x
    for block in params.encoder:
        h = _block(block, h, stride=2)

    film = apply_mlp(params.film, apply_mlp(params.cond, cond))
    gamma, beta = jnp.split(film, 2, axis=-1)
    h = h * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]

    for block in params.decoder:
        h = _block(block, _up(h), stride=1)
    return apply_conv(params.out, h, stride=1)[..., 0]
