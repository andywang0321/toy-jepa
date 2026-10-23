"""Pixel-only baseline: shared encoder, fusion, predictor, pixel head."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core.types import CNNParams, MLPParams
from .cnn import apply_cnn, init_cnn
from .mlp import apply_mlp, init_mlp


class BaselineParams(NamedTuple):
    """Parameters of the pixel-only baseline."""

    encoder: CNNParams
    fusion: MLPParams
    predictor: MLPParams
    pixel_head: MLPParams


def init_baseline(
    key: jax.Array,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    pixel_dim: int = 64,
) -> BaselineParams:
    """Initialize the encoder, fusion, predictor, and pixel head."""
    encoder_key, fusion_key, predictor_key, pixel_key = jax.random.split(key, 4)
    return BaselineParams(
        encoder=init_cnn(encoder_key, embed_dim=embed_dim),
        fusion=init_mlp(fusion_key, [2 * embed_dim, hidden_dim, embed_dim]),
        predictor=init_mlp(predictor_key, [embed_dim, hidden_dim, embed_dim]),
        pixel_head=init_mlp(pixel_key, [embed_dim, 512, pixel_dim * pixel_dim]),
    )


def apply_baseline(
    params: BaselineParams,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(pixels, c_t, z_hat)`` for a ``(B, 2, H, W)`` context."""
    z_prev = apply_cnn(params.encoder, context[:, 0][..., None])
    z_curr = apply_cnn(params.encoder, context[:, 1][..., None])
    c_t = apply_mlp(params.fusion, jnp.concatenate([z_prev, z_curr], axis=-1))
    z_hat = apply_mlp(params.predictor, c_t)
    pixels = apply_mlp(params.pixel_head, z_hat)
    side = context.shape[-1]
    return pixels.reshape(-1, side, side), c_t, z_hat


def baseline_pixel_loss(
    params: BaselineParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Pixel MSE between the prediction and the future frame."""
    context, target_pixels = batch
    pixels, _, _ = apply_baseline(params, context)
    return jnp.mean((pixels - target_pixels) ** 2)
