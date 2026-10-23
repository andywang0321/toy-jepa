"""MNIST IDX loading.

The standard MNIST download is a set of IDX files. This module reads them from
``MNIST_DIR``; it does not download anything.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import MNIST_DIR
from ..core.types import UInt8Array


def _open_binary(path: Path) -> Any:
    """Open a plain or gzipped file for binary reading."""
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _read_idx_images(path: Path) -> UInt8Array:
    """Read an IDX image file into an (N, H, W) uint8 array."""
    with _open_binary(path) as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file (magic={magic}).")
        images = np.frombuffer(file.read(), dtype=np.uint8)
    expected = count * rows * cols
    if images.size != expected:
        raise ValueError(f"{path} has {images.size} bytes, expected {expected}.")
    return images.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> UInt8Array:
    """Read an IDX label file into an (N,) uint8 array."""
    with _open_binary(path) as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file (magic={magic}).")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if labels.size != count:
        raise ValueError(f"{path} has {labels.size} labels, expected {count}.")
    return labels


def _find_pair(directory: Path, prefix: str) -> tuple[Path, Path] | None:
    """Find the image/label IDX pair for a split prefix, if present."""
    image_paths = sorted(
        path for path in directory.glob(f"{prefix}-images*") if path.is_file()
    )
    label_paths = sorted(
        path for path in directory.glob(f"{prefix}-labels*") if path.is_file()
    )
    if not image_paths or not label_paths:
        return None
    return image_paths[0], label_paths[0]


def load_mnist(mnist_dir: Path = MNIST_DIR) -> tuple[UInt8Array, UInt8Array]:
    """Load MNIST images and labels from a directory of IDX files.

    Supports the standard ``train-*`` and ``t10k-*`` files, with or without
    ``.gz``. The test split is optional. Returns images of shape (N, 28, 28)
    and labels of shape (N,).
    """
    pairs = [
        pair
        for pair in (_find_pair(mnist_dir, "train"), _find_pair(mnist_dir, "t10k"))
        if pair is not None
    ]
    if not pairs:
        raise FileNotFoundError(
            f"No MNIST IDX files found in {mnist_dir}. Expected files such as "
            "'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte'."
        )
    images = np.concatenate([_read_idx_images(path) for path, _ in pairs], axis=0)
    labels = np.concatenate([_read_idx_labels(path) for _, path in pairs], axis=0)
    return images, labels
