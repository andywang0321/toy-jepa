"""Phase-2 training and frozen-feature probe suite."""

from __future__ import annotations

import csv
import json
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import MovingMNIST
from ..data.dataset import generate_dataset
from ..evaluation.metrics import effective_rank, feature_variance, mean_pairwise_cosine
from ..models.mlp import apply_mlp
from .config import EMBED_DIM, FINAL_PROB, K_TEST, K_TRAIN, M_TEST, M_TRAIN, Config
from .data import WindowBatch, sample_batch
from .models import (
    JEPAParams,
    PixelParams,
    apply_jepa,
    apply_pixel,
    init_jepa,
    init_pixel,
    jepa_loss,
    pixel_loss,
)
from .optax_train import TrainState, make_optimizer, train_step
from .readout import (
    bounce_accuracy,
    digit_accuracy,
    fit_bounce_probe,
    fit_digit_probe,
    fit_state_probe,
    state_metrics,
)

Params = PixelParams | JEPAParams
Apply = Callable[[Any, WindowBatch], tuple[jax.Array, jax.Array]]


def _make_model(objective: str, key: jax.Array) -> tuple[Params, Callable[..., jax.Array]]:
    if objective == "pixel":
        return init_pixel(key), pixel_loss
    if objective == "jepa":
        return init_jepa(key), jepa_loss
    raise ValueError(f"unknown objective {objective!r}")


def _features(apply: Apply, params: Params, batch: WindowBatch) -> tuple[np.ndarray, np.ndarray]:
    c_t, z_hat = apply(params, batch)
    return np.asarray(c_t), np.asarray(z_hat)


def _state_features(
    apply: Apply,
    params: Params,
    dataset: MovingMNIST,
    split: int,
    count: int,
    *,
    m_values: tuple[int, ...],
    k_values: tuple[int, ...],
    final_prob: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = sample_batch(
        rng,
        dataset,
        split,
        count,
        m_values=m_values,
        k_values=k_values,
        final_prob=final_prob,
    )
    c_t, z_hat = _features(apply, params, batch)
    query = np.asarray(batch.query)
    return {
        "c_t": np.concatenate([c_t, query], axis=-1),
        "z_hat": np.concatenate([z_hat, query], axis=-1),
        "target": np.asarray(batch.target_state),
    }


def _digit_features(
    apply: Apply,
    params: Params,
    dataset: MovingMNIST,
    split: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = sample_batch(rng, dataset, split, count, final_prob=0.5)
    c_t, z_hat = _features(apply, params, batch)
    return c_t, z_hat, np.asarray(batch.digit_label)


def _bounce_features(
    apply: Apply,
    params: Params,
    dataset: MovingMNIST,
    split: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch = sample_batch(rng, dataset, split, count, final_prob=0.5)
    c_t, z_hat = _features(apply, params, batch)
    query = np.asarray(batch.query)
    return (
        np.concatenate([c_t, query], axis=-1),
        np.concatenate([z_hat, query], axis=-1),
        np.asarray(batch.bounce),
    )


def _train_model(
    config: Config,
    objective: str,
    dataset: MovingMNIST,
    key: jax.Array,
) -> tuple[Params, Apply, list[dict[str, float]]]:
    params, loss_fn = _make_model(objective, key)
    optimizer = make_optimizer(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    state = TrainState(params=params, opt_state=optimizer.init(params))
    rng = np.random.default_rng(config.seed)
    val_batches = [
        sample_batch(rng, dataset, 1, config.batch_size, final_prob=FINAL_PROB) for _ in range(config.eval_batches)
    ]

    @jax.jit
    def step(state: TrainState[Params], batch: WindowBatch):
        return train_step(state, batch, loss_fn, optimizer)

    @jax.jit
    def validation(params: Params) -> jax.Array:
        return jnp.mean(jnp.stack([loss_fn(params, batch) for batch in val_batches]))

    history: list[dict[str, float]] = []
    for step_number in range(1, config.num_steps + 1):
        batch = sample_batch(rng, dataset, 0, config.batch_size)
        state, loss = step(state, batch)
        if step_number % config.eval_every == 0 or step_number == config.num_steps:
            row = {
                "step": float(step_number),
                "train_loss": float(loss),
                "val_loss": float(validation(state.params)),
            }
            history.append(row)
            print(f"{objective} {step_number:5d} train {row['train_loss']:.5f} val {row['val_loss']:.5f}")

    if objective == "pixel":

        def apply(params: Params, batch: WindowBatch) -> tuple[jax.Array, jax.Array]:
            return apply_pixel(cast(PixelParams, params), batch)[:2]

    else:
        apply = apply_jepa
    return state.params, apply, history


def _probe_rows(
    config: Config,
    apply: Apply,
    params: Params,
    dataset: MovingMNIST,
    key: jax.Array,
) -> dict[str, float]:
    keys = jax.random.split(key, 6)
    train = _state_features(
        apply,
        params,
        dataset,
        0,
        config.readout_pairs,
        m_values=M_TRAIN,
        k_values=K_TRAIN,
        final_prob=FINAL_PROB,
        seed=config.seed + 1,
    )
    val = _state_features(
        apply,
        params,
        dataset,
        1,
        config.readout_pairs // 2,
        m_values=M_TRAIN,
        k_values=K_TRAIN,
        final_prob=FINAL_PROB,
        seed=config.seed + 2,
    )
    test_by_query: dict[str, dict[str, np.ndarray]] = {}
    for index, k in enumerate((*K_TEST, -1)):
        tag = "final" if k < 0 else f"k{k}"
        test_by_query[tag] = _state_features(
            apply,
            params,
            dataset,
            2,
            config.eval_pairs_per_k if k >= 0 else config.final_pairs,
            m_values=M_TEST,
            k_values=(k,) if k >= 0 else K_TRAIN,
            final_prob=1.0 if k < 0 else 0.0,
            seed=config.seed + 10 + index,
        )

    rows: dict[str, float] = {}
    for name in ("c_t", "z_hat"):
        result = fit_state_probe(
            keys[0 if name == "c_t" else 1],
            train[name],
            train["target"],
            val[name],
            val["target"],
            num_steps=config.readout_steps,
            batch_size=config.readout_batch_size,
            learning_rate=config.readout_lr,
            seed=config.seed,
        )
        for tag, test in test_by_query.items():
            predictions = np.asarray(apply_mlp_probe(result, test[name]))
            metrics = state_metrics(predictions, test["target"])
            for metric, value in metrics.items():
                rows[f"{name}_state_{tag}_{metric}"] = value

    digit_train = _digit_features(apply, params, dataset, 0, config.readout_pairs, config.seed + 20)
    digit_val = _digit_features(apply, params, dataset, 1, config.readout_pairs // 2, config.seed + 21)
    digit_test = _digit_features(apply, params, dataset, 2, config.readout_pairs // 2, config.seed + 22)
    for name, index in (("c_t", 0), ("z_hat", 1)):
        result = fit_digit_probe(
            keys[2 + index],
            digit_train[index],
            digit_train[2],
            digit_val[index],
            digit_val[2],
            num_steps=config.readout_steps,
            batch_size=config.readout_batch_size,
            learning_rate=config.readout_lr,
            seed=config.seed,
        )
        rows[f"{name}_digit_accuracy"] = digit_accuracy(
            np.asarray(apply_mlp_probe(result, digit_test[index])), digit_test[2]
        )

    bounce_train = _bounce_features(apply, params, dataset, 0, config.readout_pairs, config.seed + 30)
    bounce_val = _bounce_features(apply, params, dataset, 1, config.readout_pairs // 2, config.seed + 31)
    bounce_test = _bounce_features(apply, params, dataset, 2, config.readout_pairs // 2, config.seed + 32)
    for name, index in (("c_t", 0), ("z_hat", 1)):
        result = fit_bounce_probe(
            keys[4 + index],
            bounce_train[index],
            bounce_train[2],
            bounce_val[index],
            bounce_val[2],
            num_steps=config.readout_steps,
            batch_size=config.readout_batch_size,
            learning_rate=config.readout_lr,
            seed=config.seed,
        )
        rows[f"{name}_bounce_accuracy"] = bounce_accuracy(
            np.asarray(apply_mlp_probe(result, bounce_test[index])), bounce_test[2]
        )

    test_features = test_by_query["final"]
    for name in ("c_t", "z_hat"):
        features = test_features[name][:, :EMBED_DIM]
        rows[f"{name}_variance"] = feature_variance(features)
        rows[f"{name}_cosine"] = mean_pairwise_cosine(features)
        rows[f"{name}_rank"] = effective_rank(features)
    return rows


def apply_mlp_probe(result, features: np.ndarray) -> np.ndarray:
    return np.asarray(apply_mlp(result.params, jnp.asarray(features)))


def run(objective: str, config: Config) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(config.seed, config.num_samples)
    params, apply, history = _train_model(config, objective, dataset, jax.random.key(config.seed))
    probes = _probe_rows(config, apply, params, dataset, jax.random.key(config.seed + 100))

    payload = {key: value for key, value in config._asdict().items() if key != "run_dir"}
    payload.update(objective=objective, run_dir=str(config.run_dir))
    (config.run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))
    with (config.run_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("step", "train_loss", "val_loss"))
        writer.writeheader()
        writer.writerows(history)
    with (config.run_dir / "probes.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "value"))
        writer.writerows(sorted(probes.items()))
    print(f"wrote {config.run_dir}")
