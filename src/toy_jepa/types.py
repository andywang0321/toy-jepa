"""Shared array aliases and PyTree records."""

from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

import jax
import numpy as np
from numpy.typing import NDArray

from tiny_worlds.friction_mnist import FrictionMNIST

type UInt8Array = NDArray[np.uint8]
type Int8Array = NDArray[np.int8]
type Int32Array = NDArray[np.int32]
type Float32Array = NDArray[np.float32]
type ArrayLike = Float32Array | jax.Array


MovingMNIST = FrictionMNIST


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


class EMAState(NamedTuple, Generic[ParamsT]):
    """Online parameters/optimizer state plus a moving target encoder."""

    train_state: TrainState[ParamsT]
    target_encoder: CNNParams
