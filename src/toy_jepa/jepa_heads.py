"""Small generative heads conditioned on frozen JEPA c_t."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .flow_cnn import FlowCNNParams, apply_flow_cnn, init_flow_cnn
from .mlp import MLPParams, apply_mlp, init_mlp


class VAEParams(NamedTuple):
    encoder: MLPParams
    decoder: MLPParams


def _flow_cnn_loss(
    head: FlowCNNParams,
    batch: tuple[jax.Array, ...],
    key: jax.Array,
) -> jax.Array:
    c_t, query, target = batch
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, target.shape)
    time = jax.random.uniform(time_key, (target.shape[0], 1))
    x_t = (1.0 - time[..., None]) * noise + time[..., None] * target
    velocity = apply_flow_cnn(
        head,
        x_t[..., None],
        jnp.concatenate([c_t, query, time], axis=-1),
    )
    return jnp.mean((velocity - (target - noise)) ** 2)


def train_flow_cnn_head(
    c_t: jax.Array,
    query: jax.Array,
    target: jax.Array,
    key: jax.Array,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float = 1e-3,
) -> FlowCNNParams:
    head = init_flow_cnn(key, cond_dim=2 * c_t.shape[-1] + 1)
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    state = optimizer.init(head)
    rng = np.random.default_rng(int(jax.random.bits(key)))

    @jax.jit
    def step(head, state, batch, step_key):
        loss, grads = jax.value_and_grad(_flow_cnn_loss)(head, batch, step_key)
        updates, state = optimizer.update(grads, state, head)
        return optax.apply_updates(head, updates), state, loss

    train_key = key
    for step_number in range(steps):
        rows = rng.integers(0, c_t.shape[0], batch_size)
        train_key, step_key = jax.random.split(train_key)
        head, state, loss = step(
            head,
            state,
            (c_t[rows], query[rows], target[rows]),
            step_key,
        )
        if step_number % 250 == 0:
            print(f"jepa flow-CNN head {step_number:4d} loss {float(loss):.4f}")
    return head


def sample_flow_cnn_head(
    head: FlowCNNParams,
    c_t: jax.Array,
    query: jax.Array,
    key: jax.Array,
    *,
    steps: int = 24,
) -> jax.Array:
    frame = jax.random.normal(key, (c_t.shape[0], 64, 64))
    for index in range(steps):
        time = jnp.full((c_t.shape[0], 1), index / steps)
        velocity = apply_flow_cnn(
            head,
            frame[..., None],
            jnp.concatenate([c_t, query, time], axis=-1),
        )
        frame = frame + velocity / steps
    return jnp.clip(frame, 0.0, 1.0)


def _vae_loss(params: VAEParams, batch: tuple[jax.Array, ...], key: jax.Array) -> jax.Array:
    c_t, query, target = batch
    x = target.reshape(target.shape[0], -1)
    encoded = apply_mlp(params.encoder, jnp.concatenate([x, c_t, query], axis=-1))
    mu, logvar = jnp.split(encoded, 2, axis=-1)
    logvar = jnp.clip(logvar, -10.0, 10.0)
    z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
    logits = apply_mlp(params.decoder, jnp.concatenate([z, c_t, query], axis=-1))
    recon = jnp.mean((jax.nn.sigmoid(logits) - x) ** 2)
    kl = -0.5 * jnp.mean(1.0 + logvar - mu**2 - jnp.exp(logvar))
    return recon + 1e-3 * kl


def train_vae_head(
    c_t: jax.Array,
    query: jax.Array,
    target: jax.Array,
    key: jax.Array,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float = 1e-3,
) -> VAEParams:
    key_e, key_d = jax.random.split(key)
    params = VAEParams(
        encoder=init_mlp(key_e, [64 * 64 + 2 * c_t.shape[-1], 512, 64]),
        decoder=init_mlp(key_d, [32 + 2 * c_t.shape[-1], 512, 64 * 64]),
    )
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    state = optimizer.init(params)
    rng = np.random.default_rng(int(jax.random.bits(key)))

    @jax.jit
    def step(params, state, batch, step_key):
        loss, grads = jax.value_and_grad(_vae_loss)(params, batch, step_key)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    train_key = key
    for step_number in range(steps):
        rows = rng.integers(0, c_t.shape[0], batch_size)
        train_key, step_key = jax.random.split(train_key)
        params, state, loss = step(
            params,
            state,
            (c_t[rows], query[rows], target[rows]),
            step_key,
        )
        if step_number % 250 == 0:
            print(f"jepa VAE head {step_number:4d} loss {float(loss):.4f}")
    return params


def sample_vae_head(
    params: VAEParams,
    c_t: jax.Array,
    query: jax.Array,
    key: jax.Array,
) -> jax.Array:
    z = jax.random.normal(key, (c_t.shape[0], 32))
    logits = apply_mlp(params.decoder, jnp.concatenate([z, c_t, query], axis=-1))
    return jax.nn.sigmoid(logits).reshape(-1, 64, 64)
