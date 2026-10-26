"""Pixel-prediction and stop-grad JEPA models for phase 2."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .types import CNNParams, MLPParams
from .cnn import apply_cnn, init_cnn
from .mlp import apply_mlp, init_mlp
from .config import EMBED_DIM, HEADS, LAYERS, MLP_RATIO, PIXEL_HIDDEN
from .flow_cnn import FlowCNNParams, apply_flow_cnn as apply_cnn_flow, init_flow_cnn
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


class FlowMLPParams(NamedTuple):
    context: ContextParams
    velocity: MLPParams


def init_flow_mlp(
    key: jax.Array,
    *,
    embed_dim: int = EMBED_DIM,
    heads: int = HEADS,
    layers: int = LAYERS,
    mlp_ratio: int = MLP_RATIO,
    pixel_hidden: int = PIXEL_HIDDEN,
) -> FlowMLPParams:
    context_key, velocity_key = jax.random.split(key)
    return FlowMLPParams(
        context=init_context(
            context_key,
            embed_dim=embed_dim,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
        ),
        velocity=init_mlp(velocity_key, [2 * embed_dim + 1 + 64 * 64, pixel_hidden, 64 * 64]),
    )


def apply_flow_mlp(
    params: FlowMLPParams,
    batch: WindowBatch,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(c_t, x_t, velocity residual)`` for flow matching."""
    c_t, query = apply_context(params.context, batch)
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, batch.target_frame.shape)
    time = jax.random.uniform(time_key, (batch.target_frame.shape[0], 1))
    time_map = time.reshape(-1, 1, 1)
    x_t = (1.0 - time_map) * noise + time_map * batch.target_frame
    velocity = apply_mlp(
        params.velocity,
        jnp.concatenate(
            [c_t, query, time, x_t.reshape(x_t.shape[0], -1)], axis=-1
        ),
    ).reshape(-1, 64, 64)
    return c_t, x_t, velocity - (batch.target_frame - noise)


def flow_mlp_loss(params: FlowMLPParams, batch: WindowBatch, key: jax.Array) -> jax.Array:
    _, _, residual = apply_flow_mlp(params, batch, key)
    return jnp.mean(residual**2)


def sample_flow_mlp(
    params: FlowMLPParams,
    batch: WindowBatch,
    key: jax.Array,
    *,
    steps: int = 10,
) -> jax.Array:
    """Euler sample a future frame from the flow model."""
    c_t, query = apply_context(params.context, batch)
    frame = jax.random.normal(key, batch.target_frame.shape)
    for index in range(steps):
        time = jnp.full((batch.target_frame.shape[0], 1), index / steps)
        velocity = apply_mlp(
            params.velocity,
            jnp.concatenate(
                [c_t, query, time, frame.reshape(frame.shape[0], -1)], axis=-1
            ),
        ).reshape(-1, 64, 64)
        frame = frame + velocity / steps
    return jnp.clip(frame, 0.0, 1.0)


class FlowCNNModelParams(NamedTuple):
    context: ContextParams
    cnn: FlowCNNParams


def init_flow_cnn_model(
    key: jax.Array,
    *,
    embed_dim: int = EMBED_DIM,
    heads: int = HEADS,
    layers: int = LAYERS,
    mlp_ratio: int = MLP_RATIO,
) -> FlowCNNModelParams:
    context_key, cnn_key = jax.random.split(key)
    return FlowCNNModelParams(
        context=init_context(
            context_key,
            embed_dim=embed_dim,
            heads=heads,
            layers=layers,
            mlp_ratio=mlp_ratio,
        ),
        cnn=init_flow_cnn(cnn_key, cond_dim=2 * embed_dim + 1),
    )


def apply_flow_cnn(
    params: FlowCNNModelParams,
    batch: WindowBatch,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    c_t, query = apply_context(params.context, batch)
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, batch.target_frame.shape)
    time = jax.random.uniform(time_key, (batch.target_frame.shape[0], 1))
    time_map = time.reshape(-1, 1, 1)
    x_t = (1.0 - time_map) * noise + time_map * batch.target_frame
    velocity = apply_cnn_flow(
        params.cnn,
        x_t[..., None],
        jnp.concatenate([c_t, query, time], axis=-1),
    )
    return c_t, x_t, velocity - (batch.target_frame - noise)


def flow_cnn_loss(params: FlowCNNModelParams, batch: WindowBatch, key: jax.Array) -> jax.Array:
    _, _, residual = apply_flow_cnn(params, batch, key)
    return jnp.mean(residual**2)


def sample_flow_cnn(
    params: FlowCNNModelParams,
    batch: WindowBatch,
    key: jax.Array,
    *,
    steps: int = 24,
) -> jax.Array:
    c_t, query = apply_context(params.context, batch)
    frame = jax.random.normal(key, batch.target_frame.shape)
    for index in range(steps):
        time = jnp.full((batch.target_frame.shape[0], 1), index / steps)
        velocity = apply_cnn_flow(
            params.cnn,
            frame[..., None],
            jnp.concatenate([c_t, query, time], axis=-1),
        )
        frame = frame + velocity / steps
    return jnp.clip(frame, 0.0, 1.0)
