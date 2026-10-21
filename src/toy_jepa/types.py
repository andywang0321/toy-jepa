"""Shared type aliases and dataset records.

Kept separate from ``constants.py`` so the project vocabulary lives in one
place. Import this module relatively (``from .types import ...``) so it does not
shadow the standard-library ``types`` module.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

UInt8Array: TypeAlias = NDArray[np.uint8]
Int8Array: TypeAlias = NDArray[np.int8]
Int32Array: TypeAlias = NDArray[np.int32]
Float32Array: TypeAlias = NDArray[np.float32]


class MovingMNIST(NamedTuple):
    """A labeled Moving MNIST dataset. NamedTuple is better than dataclass for simple applications."""

    frames: UInt8Array # (N, T, H, W) uint8 frames.
    digit_label: Int8Array # (N, 2) int8 MNIST class labels in [0, 9].
    digit_index: Int32Array # (N, 2) int32 indices into the source MNIST images.
    position: Float32Array # (N, T, 2, 2) float32 top-left (x, y) per digit, in pixels.
    velocity: Float32Array # (N, T, 2, 2) float32 (vx, vy) per digit, in pixels/frame.
    split: Int8Array # (N,) int8 split id: 0=train, 1=val, 2=test.
