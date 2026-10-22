"""Exponential moving average of the JEPA target encoder."""

from __future__ import annotations

import jax

from .types import CNNParams


def ema_update(
    target: CNNParams, online: CNNParams, decay: float
) -> CNNParams:
    """Move ``target`` toward ``online``: ``decay * target + (1-decay) * online``."""
    return jax.tree.map(
        lambda target_leaf, online_leaf: decay * target_leaf
        + (1.0 - decay) * online_leaf,
        target,
        online,
    )
