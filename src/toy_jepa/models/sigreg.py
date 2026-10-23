"""SIGReg: sketch isotropic Gaussian regularization (LeJEPA §3).

Compares the empirical characteristic function of random 1-d projections of
the features against the standard-normal characteristic function. A collapsed
feature cloud produces a characteristic function that is far from the target,
so the statistic is positive; well-spread features keep it small. The loss is
purely geometric: it does not predict a target and needs no labels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _trapezoid_weights(num_points: int, t_max: float) -> tuple[jax.Array, jax.Array]:
    """Return the quadrature grid and its composite trapezoid weights."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive.")

    t = jnp.linspace(0.0, t_max, num_points)
    width = t_max / (num_points - 1)
    weights = jnp.full((num_points,), width)
    weights = weights.at[0].set(0.5 * width)
    weights = weights.at[-1].set(0.5 * width)
    return t, weights


def sigreg_loss(
    features: jax.Array,
    key: jax.Array,
    *,
    num_projections: int = 256,
    num_points: int = 17,
    t_max: float = 3.0,
) -> jax.Array:
    """SIGReg loss for a ``(batch, dim)`` feature matrix.

    For random unit directions ``u_m`` and a trapezoid grid ``t_j`` with
    weights ``w_j``, the statistic is::

        phi_m(t) = mean_b exp(i t (c_b . u_m))
        loss = mean_m sum_j w_j |phi_m(t_j) - exp(-t_j^2 / 2)|^2

    The real and imaginary parts are computed separately with ``cos``/``sin``,
    so no complex dtype is needed.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape (batch, dim).")
    if features.shape[0] == 0:
        raise ValueError("features must contain at least one sample.")

    dim = features.shape[1]
    directions = jax.random.normal(key, (dim, num_projections))
    directions = directions / jnp.linalg.norm(directions, axis=0, keepdims=True)
    projections = features @ directions  # (B, M)

    t, weights = _trapezoid_weights(num_points, t_max)
    angles = projections[..., None] * t  # (B, M, J)

    real = jnp.mean(jnp.cos(angles), axis=0)  # (M, J)
    imag = jnp.mean(jnp.sin(angles), axis=0)  # (M, J)
    target = jnp.exp(-0.5 * t**2)  # (J,)

    mismatch = (real - target) ** 2 + imag**2
    return jnp.mean(jnp.sum(mismatch * weights, axis=-1))
