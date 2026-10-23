"""Barebones JEPA: shared encoder, no target asymmetry."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..types import CNNParams, MLPParams
from .cnn import apply_cnn, init_cnn
from .mlp import apply_mlp, init_mlp


class JEPABaseParams(NamedTuple):
    """Parameters of the barebones JEPA model."""

    encoder: CNNParams
    fusion: MLPParams
    predictor: MLPParams


def init_jepa_base(
    key: jax.Array,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 256,
) -> JEPABaseParams:
    """Initialize the frame encoder, context fusion, and predictor."""
    encoder_key, fusion_key, predictor_key = jax.random.split(key, 3)
    return JEPABaseParams(
        encoder=init_cnn(encoder_key, embed_dim=embed_dim),
        fusion=init_mlp(fusion_key, [2 * embed_dim, hidden_dim, embed_dim]),
        predictor=init_mlp(predictor_key, [embed_dim, hidden_dim, embed_dim]),
    )


def apply_jepa_base(
    params: JEPABaseParams,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(z_hat, c_t)`` for a ``(B, 2, H, W)`` context."""
    z_prev = apply_cnn(params.encoder, context[:, 0][..., None])
    z_curr = apply_cnn(params.encoder, context[:, 1][..., None])
    c_t = apply_mlp(params.fusion, jnp.concatenate([z_prev, z_curr], axis=-1))
    return apply_mlp(params.predictor, c_t), c_t


def jepa_base_loss(
    params: JEPABaseParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Latent MSE with gradients through both the online and target paths."""
    context, target_frame = batch
    z_hat, _ = apply_jepa_base(params, context)
    z_target = apply_cnn(params.encoder, target_frame[..., None])
    return jnp.mean((z_hat - z_target) ** 2)
