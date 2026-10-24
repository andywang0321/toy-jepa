"""Tests for the hand-written AdamW optimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from toy_jepa.core.optim import adamw_init, adamw_update


def test_adamw_init_matches_param_structure() -> None:
    """Moments must mirror the parameter PyTree, with the same shapes/dtypes."""
    params = [
        {"w": jnp.ones((2, 3)), "b": jnp.zeros(3)},
        {"w": jnp.ones((3, 1)), "b": jnp.zeros(1)},
    ]
    state = adamw_init(params)

    assert int(state.t) == 0
    assert state.t.shape == ()
    assert state.t.dtype == jnp.int32
    assert jax.tree.structure(state.m) == jax.tree.structure(params)
    assert jax.tree.structure(state.v) == jax.tree.structure(params)

    for param, m, v in zip(
        jax.tree.leaves(params),
        jax.tree.leaves(state.m),
        jax.tree.leaves(state.v),
    ):
        assert m.shape == param.shape
        assert v.shape == param.shape
        assert m.dtype == param.dtype
        assert v.dtype == param.dtype
        assert jnp.all(m == 0)
        assert jnp.all(v == 0)


def test_adamw_bias_correction_first_step() -> None:
    """At t=1, m_hat = g and v_hat = g**2, so the Adam step is sign(g)."""
    params = {"w": jnp.array([0.0])}
    grads = {"w": jnp.array([2.0])}
    state = adamw_init(params)

    params_next, state_next = adamw_update(params, grads, state, lr=0.1, weight_decay=0.0)

    assert jnp.allclose(params_next["w"], jnp.array([-0.1]), atol=1e-6)
    assert int(state_next.t) == 1
    assert jnp.allclose(state_next.m["w"], jnp.array([0.2]))
    assert jnp.allclose(state_next.v["w"], jnp.array([0.004]))


def test_adamw_decoupled_weight_decay() -> None:
    """With a zero gradient, only the decoupled weight decay changes the param."""
    params = {"w": jnp.array([2.0])}
    grads = {"w": jnp.zeros_like(params["w"])}
    state = adamw_init(params)

    params_next, _ = adamw_update(params, grads, state, lr=0.1, weight_decay=0.1)

    assert jnp.allclose(params_next["w"], jnp.array([2.0 * (1.0 - 0.1 * 0.1)]), atol=1e-6)


def test_adamw_converges_on_quadratic() -> None:
    """Minimize 0.5 * (w - 3)**2 from w=0 with a jitted AdamW step."""
    target = jnp.array([3.0])
    params = {"w": jnp.array([0.0])}
    state = adamw_init(params)

    def loss_fn(p):
        return 0.5 * jnp.sum((p["w"] - target) ** 2)

    @jax.jit
    def step(p, s):
        grads = jax.grad(loss_fn)(p)
        return adamw_update(p, grads, s, lr=0.1, weight_decay=0.0)

    for _ in range(500):
        params, state = step(params, state)

    assert jnp.allclose(params["w"], target, atol=1e-3)
    assert int(state.t) == 500
    assert loss_fn(params) < 1e-6
