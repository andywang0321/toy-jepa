"""Pixel-prediction and stop-grad JEPA models for phase 2."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..core.types import CNNParams, MLPParams
from ..models.cnn import apply_cnn, init_cnn
from ..models.mlp import apply_mlp, init_mlp
from .config import EMBED_DIM, HEADS, LAYERS, MLP_RATIO, PIXEL_HIDDEN
from .data import WindowBatch
from .transformer import TemporalParams, apply_temporal, init_temporal


class ContextParams(NamedTuple):
    encoder: CNNParams
    temporal: TemporalParams
    query: MLPParams


class PixelParams(NamedTuple):
    context: ContextParams
    decoder: MLPParams


class JEPAParams(NamedTuple):
    context: ContextParams
    predictor: MLPParams


def init_context(
    key: jax.Array,
    *,
    embed_dim: int = EMBED_DIM,
    heads: int = HEADS,
    layers: int = LAYERS,
    mlp_ratio: int = MLP_RATIO,
) -> ContextParams:
    encoder_key, temporal_key, query_key = jax.random.split(key, 3)
    return ContextParams(
        encoder=init_cnn(encoder_key, embed_dim=embed_dim),
        temporal=init_temporal(
            temporal_key,
            dim=embed_dim,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
        ),
        query=init_mlp(query_key, [2, embed_dim]),
    )


def apply_context(
    params: ContextParams,
    batch: WindowBatch,
    *,
    heads: int = HEADS,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(c_t, query_embedding)``."""
    batch_size, length, height, width = batch.context.shape
    frames = batch.context.reshape(batch_size * length, height, width, 1)
    tokens = apply_cnn(params.encoder, frames).reshape(batch_size, length, -1)
    c_t = apply_temporal(params.temporal, tokens, batch.mask, heads=heads)
    return c_t, apply_mlp(params.query, batch.query)


def init_pixel(
    key: jax.Array,
    *,
    embed_dim: int = EMBED_DIM,
    heads: int = HEADS,
    layers: int = LAYERS,
    mlp_ratio: int = MLP_RATIO,
    pixel_hidden: int = PIXEL_HIDDEN,
) -> PixelParams:
    context_key, decoder_key = jax.random.split(key)
    return PixelParams(
        context=init_context(
            context_key,
            embed_dim=embed_dim,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
        ),
        decoder=init_mlp(decoder_key, [2 * embed_dim, pixel_hidden, 64 * 64]),
    )


def apply_pixel(
    params: PixelParams,
    batch: WindowBatch,
    *,
    heads: int = HEADS,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(c_t, encoded_prediction, predicted_frame)``."""
    c_t, query = apply_context(params.context, batch, heads=heads)
    logits = apply_mlp(params.decoder, jnp.concatenate([c_t, query], axis=-1))
    frame = jax.nn.sigmoid(logits).reshape(-1, 64, 64)
    z_hat = apply_cnn(params.context.encoder, frame[..., None])
    return c_t, z_hat, frame


def pixel_loss(params: PixelParams, batch: WindowBatch) -> jax.Array:
    _, _, frame = apply_pixel(params, batch)
    return jnp.mean((frame - batch.target_frame) ** 2)


def init_jepa(
    key: jax.Array,
    *,
    embed_dim: int = EMBED_DIM,
    heads: int = HEADS,
    layers: int = LAYERS,
    mlp_ratio: int = MLP_RATIO,
) -> JEPAParams:
    context_key, predictor_key = jax.random.split(key)
    return JEPAParams(
        context=init_context(
            context_key,
            embed_dim=embed_dim,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
        ),
        predictor=init_mlp(predictor_key, [2 * embed_dim, 256, embed_dim]),
    )


def apply_jepa(
    params: JEPAParams,
    batch: WindowBatch,
    *,
    heads: int = HEADS,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(c_t, z_hat)``."""
    c_t, query = apply_context(params.context, batch, heads=heads)
    return c_t, apply_mlp(params.predictor, jnp.concatenate([c_t, query], axis=-1))


def jepa_loss(params: JEPAParams, batch: WindowBatch) -> jax.Array:
    _, z_hat = apply_jepa(params, batch)
    z_target = apply_cnn(params.context.encoder, batch.target_frame[..., None])
    return jnp.mean((z_hat - jax.lax.stop_gradient(z_target)) ** 2)
