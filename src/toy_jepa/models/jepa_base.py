"""Barebones JEPA architecture.

z_prev = encoder(x_{t-1})
z_curr = encoder(x_t)
c_t    = fuse(z_prev, z_curr)
z_hat  = predictor(c_t)

The target latent, ``encoder(x_{t+k})``, is computed by the loss so the
target branch can later change independently (stop-grad in M3, EMA in M4).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..types import JEPABaseParams
from .cnn import apply_cnn, init_cnn
from .mlp import apply_mlp, init_mlp


def init_jepa_base(
    key: jax.Array,
    *,
    embed_dim: int = 128,
    hidden_dim: int = 256,
) -> JEPABaseParams:
    """Initialize the frame encoder, context fusion, and predictor."""
    encoder_key, fusion_key, predictor_key = jax.random.split(key, 3)
    encoder = init_cnn(encoder_key, embed_dim=embed_dim)
    fusion = init_mlp(fusion_key, [2 * embed_dim, hidden_dim, embed_dim])
    predictor = init_mlp(predictor_key, [embed_dim, hidden_dim, embed_dim])
    return JEPABaseParams(
        encoder=encoder,
        fusion=fusion,
        predictor=predictor,
    )


def apply_jepa_base(
    params: JEPABaseParams,
    context: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the predicted future latent and the fused context feature."""
    z_prev = apply_cnn(params.encoder, context[:, 0][..., None])
    z_curr = apply_cnn(params.encoder, context[:, 1][..., None])
    c_t = apply_mlp(
        params.fusion, jnp.concatenate([z_prev, z_curr], axis=-1)
    )
    return apply_mlp(params.predictor, c_t), c_t
