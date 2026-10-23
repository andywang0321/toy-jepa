"""Output shapes and gradient sanity for every JEPA model."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
import pytest

from toy_jepa.models import jepa_base, jepa_ema, jepa_sigreg, jepa_stopgrad

MODELS = (
    (jepa_base.init_jepa_base, jepa_base.apply_jepa_base),
    (jepa_stopgrad.init_jepa_stopgrad, jepa_stopgrad.apply_jepa_stopgrad),
    (jepa_ema.init_jepa_ema, jepa_ema.apply_jepa_ema),
    (jepa_sigreg.init_jepa_sigreg, jepa_sigreg.apply_jepa_sigreg),
)


def _context() -> jax.Array:
    return jnp.zeros((2, 2, 64, 64), dtype=jnp.float32)


ParamsT = TypeVar("ParamsT")


def _assert_finite_grads(loss: Callable[[ParamsT], jax.Array], params: ParamsT) -> None:
    grads = jax.grad(loss)(params)
    for grad in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(grad))


@pytest.mark.parametrize("init,apply", MODELS)
def test_jepa_model_output_shapes(
    init: Callable[..., object], apply: Callable[..., tuple[jax.Array, jax.Array]]
) -> None:
    params = init(jax.random.key(0), embed_dim=32, hidden_dim=32)

    z_hat, c_t = apply(params, _context())

    assert z_hat.shape == (2, 32)
    assert c_t.shape == (2, 32)
    assert jnp.all(jnp.isfinite(z_hat))
    assert jnp.all(jnp.isfinite(c_t))


def test_base_and_stopgrad_have_finite_gradients() -> None:
    context = _context()
    target = jnp.zeros((2, 64, 64), dtype=jnp.float32)
    batch = (context, target)

    base = jepa_base.init_jepa_base(jax.random.key(0), embed_dim=32, hidden_dim=32)
    stopgrad = jepa_stopgrad.init_jepa_stopgrad(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )
    _assert_finite_grads(lambda p: jepa_base.jepa_base_loss(p, batch), base)
    _assert_finite_grads(lambda p: jepa_stopgrad.jepa_stopgrad_loss(p, batch), stopgrad)


def test_ema_and_sigreg_have_finite_gradients() -> None:
    batch = (_context(), jnp.zeros((2, 64, 64), dtype=jnp.float32))
    ema = jepa_ema.init_jepa_ema(jax.random.key(0), embed_dim=32, hidden_dim=32)
    sigreg = jepa_sigreg.init_jepa_sigreg(
        jax.random.key(0), embed_dim=32, hidden_dim=32
    )

    _assert_finite_grads(lambda p: jepa_ema.jepa_ema_loss(p, p.encoder, batch), ema)
    _assert_finite_grads(
        lambda p: jepa_sigreg.jepa_sigreg_loss(
            p, p.encoder, batch, key=jax.random.key(1), weight=0.1
        ),
        sigreg,
    )
