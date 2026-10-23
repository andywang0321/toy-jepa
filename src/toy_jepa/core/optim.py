"""A hand-written functional AdamW optimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import AdamWState, ParamsT


def adamw_init(model_params: ParamsT) -> AdamWState[ParamsT]:
    """Create zero-filled AdamW moments with the same tree as ``model_params``."""
    return AdamWState(
        t=jnp.array(0, dtype=jnp.int32),
        m=jax.tree.map(jnp.zeros_like, model_params),
        v=jax.tree.map(jnp.zeros_like, model_params),
    )


def _update_moment(prev: jax.Array, new: jax.Array, beta: float) -> jax.Array:
    return beta * prev + (1.0 - beta) * new


def adamw_update(
    params: ParamsT,
    grads: ParamsT,
    opt_state: AdamWState[ParamsT],
    *,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[ParamsT, AdamWState[ParamsT]]:
    """One AdamW update step."""
    beta1, beta2 = betas
    t = opt_state.t + 1

    m_next = jax.tree.map(lambda m, g: _update_moment(m, g, beta1), opt_state.m, grads)
    v_next = jax.tree.map(
        lambda v, g: _update_moment(v, jnp.square(g), beta2), opt_state.v, grads
    )

    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t

    def apply_leaf(param: jax.Array, m: jax.Array, v: jax.Array) -> jax.Array:
        m_hat = m / bc1
        v_hat = v / bc2
        update = m_hat / (jnp.sqrt(v_hat) + eps)
        return param * (1.0 - lr * weight_decay) - lr * update

    params_next = jax.tree.map(apply_leaf, params, m_next, v_next)
    return params_next, AdamWState(t=t, m=m_next, v=v_next)
