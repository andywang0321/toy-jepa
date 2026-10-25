"""Tests for phase-2 checkpoint saving."""

from __future__ import annotations

import jax

from toy_jepa.models.mlp import init_mlp
from toy_jepa.phase2.checkpoint import load_params, save_params


def test_checkpoint_round_trip(tmp_path) -> None:
    params = init_mlp(jax.random.key(0), [4, 8, 2])
    path = tmp_path / "params.pkl"
    save_params(params, path)
    loaded = load_params(path)

    assert len(loaded.blocks) == len(params.blocks)
    assert loaded.blocks[0].w.shape == params.blocks[0].w.shape
