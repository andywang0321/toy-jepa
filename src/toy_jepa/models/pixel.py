"""Pixel-future baseline model.

The model encodes two context frames with a shared frame encoder, fuses the
features, predicts a latent, and decodes that latent to a future frame.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import PixelParams
from .cnn import apply_cnn, init_cnn
from .mlp import apply_mlp, init_mlp


def init_pixel_model(
    key: jax.Array,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    pixel_dim: int = 64,
) -> PixelParams:
    """Initialize the frame encoder, fusion, predictor, and pixel head."""
    encoder_key, fusion_key, predictor_key, head_key = jax.random.split(key, 4)

    encoder = init_cnn(encoder_key, embed_dim=embed_dim)
    fusion = init_mlp(fusion_key, [2 * embed_dim, hidden_dim, embed_dim])
    predictor = init_mlp(predictor_key, [embed_dim, hidden_dim, embed_dim])
    pixel_head = init_mlp(head_key, [embed_dim, 512, pixel_dim * pixel_dim])

    return PixelParams(
        encoder=encoder,
        fusion=fusion,
        predictor=predictor,
        pixel_head=pixel_head,
    )


def apply_pixel_model(
    params: PixelParams,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Predict a future frame from two context frames.

    Args:
        params: pixel-model parameters.
        context: ``(B, 2, H, W)`` context frames.

    Returns:
        ``(pixels, z_context, z_hat)`` with shapes
        ``(B, H, W)``, ``(B, embed_dim)``, ``(B, embed_dim)``.
    """
    z_prev = apply_cnn(params.encoder, context[:, 0][..., None])
    z_curr = apply_cnn(params.encoder, context[:, 1][..., None])

    z_context = apply_mlp(
        params.fusion, jnp.concatenate([z_prev, z_curr], axis=-1)
    )
    z_hat = apply_mlp(params.predictor, z_context)
    pixels = apply_mlp(params.pixel_head, z_hat)

    side = context.shape[-1]
    return pixels.reshape(-1, side, side), z_context, z_hat
