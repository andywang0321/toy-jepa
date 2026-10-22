"""Baseline model with pixel and state output heads.

The baseline encodes two context frames with a shared CNN, fuses the features,
predicts a latent, and decodes it through two heads: a pixel head for the future
frame and a state head for the future state of both digits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import BaselineParams
from .cnn import apply_cnn, init_cnn
from .mlp import apply_mlp, init_mlp


def init_baseline(
    key: jax.Array,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 256,
    pixel_dim: int = 64,
    state_dim: int = 8,
) -> BaselineParams:
    """Initialize the encoder, fusion, predictor, pixel head, and state head."""
    encoder_key, fusion_key, predictor_key, pixel_key, state_key = (
        jax.random.split(key, 5)
    )

    encoder = init_cnn(encoder_key, embed_dim=embed_dim)
    fusion = init_mlp(fusion_key, [2 * embed_dim, hidden_dim, embed_dim])
    predictor = init_mlp(predictor_key, [embed_dim, hidden_dim, embed_dim])
    pixel_head = init_mlp(pixel_key, [embed_dim, 512, pixel_dim * pixel_dim])
    state_head = init_mlp(state_key, [embed_dim, hidden_dim, state_dim])

    return BaselineParams(
        encoder=encoder,
        fusion=fusion,
        predictor=predictor,
        pixel_head=pixel_head,
        state_head=state_head,
    )


def apply_baseline(
    params: BaselineParams,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Predict a future frame and future state from two context frames.

    Args:
        params: baseline parameters.
        context: ``(B, 2, H, W)`` context frames.

    Returns:
        ``(pixels, state, z_context, z_hat)`` with shapes
        ``(B, H, W)``, ``(B, 8)``, ``(B, embed_dim)``, ``(B, embed_dim)``.
    """
    z_prev = apply_cnn(params.encoder, context[:, 0][..., None])
    z_curr = apply_cnn(params.encoder, context[:, 1][..., None])

    z_context = apply_mlp(
        params.fusion, jnp.concatenate([z_prev, z_curr], axis=-1)
    )
    z_hat = apply_mlp(params.predictor, z_context)
    pixels = apply_mlp(params.pixel_head, z_hat)
    state = apply_mlp(params.state_head, z_hat)

    side = context.shape[-1]
    return pixels.reshape(-1, side, side), state, z_context, z_hat
