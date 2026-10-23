"""JEPA with a stop-gradient on the shared-encoder target."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core.types import CNNParams, MLPParams
from .cnn import apply_cnn, init_cnn
from .mlp import apply_mlp, init_mlp


class JEPAStopGradParams(NamedTuple):
    """Parameters of the stop-gradient JEPA model."""

    encoder: CNNParams
    fusion: MLPParams
    predictor: MLPParams


def init_jepa_stopgrad(
    key: jax.Array,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 256,
) -> JEPAStopGradParams:
    """Initialize the frame encoder, context fusion, and predictor."""
    encoder_key, fusion_key, predictor_key = jax.random.split(key, 3)
    return JEPAStopGradParams(
        encoder=init_cnn(encoder_key, embed_dim=embed_dim),
        fusion=init_mlp(fusion_key, [2 * embed_dim, hidden_dim, embed_dim]),
        predictor=init_mlp(predictor_key, [embed_dim, hidden_dim, embed_dim]),
    )


def apply_jepa_stopgrad(
    params: JEPAStopGradParams,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(z_hat, c_t)`` for a ``(B, 2, H, W)`` context."""
    z_prev = apply_cnn(params.encoder, context[:, 0][..., None])
    z_curr = apply_cnn(params.encoder, context[:, 1][..., None])
    c_t = apply_mlp(params.fusion, jnp.concatenate([z_prev, z_curr], axis=-1))
    return apply_mlp(params.predictor, c_t), c_t


def jepa_stopgrad_loss(
    params: JEPAStopGradParams,
    batch: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Latent MSE with ``stop_gradient`` on the shared-encoder target."""
    context, target_frame = batch
    z_hat, _ = apply_jepa_stopgrad(params, context)
    z_target = apply_cnn(params.encoder, target_frame[..., None])
    return jnp.mean((z_hat - jax.lax.stop_gradient(z_target)) ** 2)
