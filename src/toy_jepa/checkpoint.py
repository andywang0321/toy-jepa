"""Local phase-2 checkpoints (gitignored)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np

_LEGACY_CLASSES = {
    ("toy_jepa.models", "FlowParams"): ("toy_jepa.models", "FlowMLPParams"),
    ("toy_jepa.phase2.models", "FlowParams"): ("toy_jepa.models", "FlowMLPParams"),
    ("toy_jepa.phase2.models", "PixelParams"): ("toy_jepa.models", "PixelParams"),
    ("toy_jepa.phase2.models", "JEPAParams"): ("toy_jepa.models", "JEPAParams"),
    ("toy_jepa.phase2.models", "FlowCNNModelParams"): (
        "toy_jepa.models",
        "FlowCNNModelParams",
    ),
}


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        module, name = _LEGACY_CLASSES.get((module, name), (module, name))
        return super().find_class(module, name)


def save_params(params: Any, path: Path) -> None:
    """Save a param PyTree as NumPy arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = jax.tree_util.tree_map(np.asarray, params)
    with path.open("wb") as file:
        pickle.dump(arrays, file)


def load_params(path: Path) -> Any:
    with path.open("rb") as file:
        return _CheckpointUnpickler(file).load()
