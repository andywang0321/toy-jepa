"""Shared type aliases and records.

Kept separate from ``constants.py`` so the project vocabulary lives in one
place. Import this module relatively (``from .types import ...``) so it does not
shadow the standard-library ``types`` module.

Convention:
- ``type`` aliases for array/container vocabulary.
- ``NamedTuple`` records for parameter trees, dataset rows, and training state.
"""

from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

import jax
import numpy as np
from numpy.typing import NDArray

type UInt8Array = NDArray[np.uint8]
type Int8Array = NDArray[np.int8]
type Int32Array = NDArray[np.int32]
type Float32Array = NDArray[np.float32]


class MovingMNIST(NamedTuple):
    """A labeled Moving MNIST dataset."""
    frames: UInt8Array  # (N, T, H, W) uint8 frames.
    digit_label: Int8Array  # (N, 2) int8 MNIST class labels in [0, 9].
    digit_index: Int32Array  # (N, 2) int32 indices into the source images.
    position: Float32Array  # (N, T, 2, 2) top-left (x, y), pixels.
    velocity: Float32Array  # (N, T, 2, 2) (vx, vy), pixels/frame.
    split: Int8Array  # (N,) split id: 0=train, 1=val, 2=test.


# General array-leaf PyTree; model parameters use the concrete records below.
type PyTree = jax.Array | list[PyTree] | tuple[PyTree, ...] | dict[str, PyTree]


class AffineParams(NamedTuple):
    """A ``w x + b`` operation, shared by linear and convolutional layers."""
    w: jax.Array
    b: jax.Array


class NormParams(NamedTuple):
    """Affine parameters for a normalization layer."""
    gamma: jax.Array
    beta: jax.Array


class MLPParams(NamedTuple):
    """Parameters of a stack of affine layers."""
    blocks: list[AffineParams]


class ConvBlockParams(NamedTuple):
    """One ``conv -> norm`` block of the frame encoder."""
    conv: AffineParams
    norm: NormParams


class CNNParams(NamedTuple):
    """Parameters of the convolutional frame encoder."""
    blocks: list[ConvBlockParams]
    fc: MLPParams
    ln: NormParams


class BaselineParams(NamedTuple):
    """Parameters of the baseline with pixel and state heads."""
    encoder: CNNParams
    fusion: MLPParams
    predictor: MLPParams
    pixel_head: MLPParams
    state_head: MLPParams


class JEPABaseParams(NamedTuple):
    """Parameters of the barebones JEPA model."""
    encoder: CNNParams
    fusion: MLPParams
    predictor: MLPParams


# Optimizer state is generic over whatever concrete parameter record a model uses.
ParamsT = TypeVar("ParamsT")


class AdamWState(NamedTuple, Generic[ParamsT]):
    """AdamW moments, mirroring the structure of the model parameters."""
    t: jax.Array
    m: ParamsT
    v: ParamsT


class TrainState(NamedTuple, Generic[ParamsT]):
    """Parameters plus the optimizer state."""
    params: ParamsT
    opt_state: AdamWState[ParamsT]
