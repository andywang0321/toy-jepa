"""Local phase-2 checkpoints (gitignored)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np


def save_params(params: Any, path: Path) -> None:
    """Save a param PyTree as NumPy arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = jax.tree_util.tree_map(np.asarray, params)
    with path.open("wb") as file:
        pickle.dump(arrays, file)


def load_params(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)
