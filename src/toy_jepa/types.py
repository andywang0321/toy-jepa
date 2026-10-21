"""Shared type aliases and dataset records.

Kept separate from ``constants.py`` so the project vocabulary lives in one
place. Import this module relatively (``from .types import ...``) so it does not
shadow the standard-library ``types`` module.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

import jax

type UInt8Array = NDArray[np.uint8]
type Int8Array = NDArray[np.int8]
type Int32Array = NDArray[np.int32]
type Float32Array = NDArray[np.float32]


class MovingMNIST(NamedTuple):
    """A labeled Moving MNIST dataset. NamedTuple is better than dataclass for simple applications."""
    frames: UInt8Array # (N, T, H, W) uint8 frames.
    digit_label: Int8Array # (N, 2) int8 MNIST class labels in [0, 9].
    digit_index: Int32Array # (N, 2) int32 indices into the source MNIST images.
    position: Float32Array # (N, T, 2, 2) float32 top-left (x, y) per digit, in pixels.
    velocity: Float32Array # (N, T, 2, 2) float32 (vx, vy) per digit, in pixels/frame.
    split: Int8Array # (N,) int8 split id: 0=train, 1=val, 2=test.


# A PyTree whose leaves are JAX arrays (dict/list/tuple containers).
type PyTree = (
    jax.Array | list[PyTree] | tuple[PyTree, ...] | dict[str, PyTree]
)

class AdamWState(NamedTuple):
    """State of the AdamW optimizer."""
    t: jax.Array    # timestep
    m: PyTree    # 1st moment vector (grad moving avg)
    v: PyTree    # 2nd moment vector (grad**2 moving avg)

class TrainState(NamedTuple):
    params: PyTree
    opt_state: AdamWState
