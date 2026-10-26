"""Phase-2 training and frozen-feature probe suite."""

from __future__ import annotations

import csv
import json
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np

from tiny_worlds.friction_mnist import FrictionMNIST, generate_dataset
from .metrics import effective_rank, feature_variance, mean_pairwise_cosine
from .mlp import apply_mlp
from .config import EMBED_DIM, FINAL_PROB, K_TEST, K_TRAIN, M_TEST, M_TRAIN, MNIST_DIR, Config
from .checkpoint import save_params
from .data import WindowBatch, sample_batch
from .features import bounce_features, digit_features, state_features
from .models import (
    FlowParams,
    JEPAParams,
    PixelParams,
    apply_context,
    apply_jepa,
    flow_context,
    flow_loss,
    init_flow,
    init_jepa,
    init_pixel,
    jepa_loss,
    pixel_loss,
)
from .train import TrainState, make_optimizer, train_step, train_step_with_key
from .probes import (
    bounce_metrics,
    digit_accuracy,
    fit_bounce_probe,
    fit_digit_probe,
    fit_state_probe,
    state_metrics,
)

Params = PixelParams | JEPAParams | FlowParams
Apply = Callable[[Any, WindowBatch], Any]


def _make_model(objective: str, key: jax.Array) -> tuple[Params, Callable[..., jax.Array]]:
    if objective == "pixel":
        return init_pixel(key), pixel_loss
    if objective == "pixel_fm":
        return init_flow(key), flow_loss
    if objective == "jepa":
        return init_jepa(key), jepa_loss
    raise ValueError(f"unknown objective {objective!r}")


def _train_model(
    config: Config,
    objective: str,
    dataset: FrictionMNIST,
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
    is_flow = objective == "pixel_fm"
    train_key = jax.random.key(config.seed + 7)
    val_keys = tuple(jax.random.split(train_key, len(val_batches)))

    if is_flow:

        @jax.jit
        def step(state: TrainState[Params], batch: WindowBatch, key: jax.Array):
            return train_step_with_key(state, batch, key, loss_fn, optimizer)

        @jax.jit
        def validation(params: Params) -> jax.Array:
            return jnp.mean(
                jnp.stack([loss_fn(params, batch, key) for batch, key in zip(val_batches, val_keys)])
            )

    else:

        @jax.jit
        def step(state: TrainState[Params], batch: WindowBatch):
            return train_step(state, batch, loss_fn, optimizer)

        @jax.jit
        def validation(params: Params) -> jax.Array:
            return jnp.mean(jnp.stack([loss_fn(params, batch) for batch in val_batches]))

    history: list[dict[str, float]] = []
    for step_number in range(1, config.num_steps + 1):
        batch = sample_batch(rng, dataset, 0, config.batch_size)
        if is_flow:
            train_key, step_key = jax.random.split(train_key)
            state, loss = step(state, batch, step_key)
        else:
            state, loss = step(state, batch)
        if step_number % config.eval_every == 0 or step_number == config.num_steps:
            row = {
                "step": float(step_number),
                "train_loss": float(loss),
                "val_loss": float(validation(state.params)),
            }
            history.append(row)
            print(f"{objective} {step_number:5d} train {row['train_loss']:.5f} val {row['val_loss']:.5f}")

    def pixel_context(params: Params, batch: WindowBatch) -> Any:
        return apply_context(cast(PixelParams, params).context, batch)[0]

    def flow_c_t(params: Params, batch: WindowBatch) -> Any:
        return flow_context(cast(FlowParams, params), batch)

    def jepa_context(params: Params, batch: WindowBatch) -> Any:
        return apply_jepa(cast(JEPAParams, params), batch)[0]

    if objective == "pixel":
        apply: Apply = pixel_context
    elif objective == "pixel_fm":
        apply = flow_c_t
    else:
        apply = jepa_context
    return state.params, apply, history


def _probe_rows(
    config: Config,
    apply: Apply,
    params: Params,
    dataset: FrictionMNIST,
    key: jax.Array,
) -> dict[str, float]:
    keys = jax.random.split(key, 3)
    train = state_features(
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
    val = state_features(
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
        test_by_query[tag] = state_features(
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
    result = fit_state_probe(
        keys[0],
        train["c_t"],
        train["target"],
        val["c_t"],
        val["target"],
        num_steps=config.readout_steps,
        batch_size=config.readout_batch_size,
        learning_rate=config.readout_lr,
        seed=config.seed,
    )
    for tag, test in test_by_query.items():
        predictions = np.asarray(apply_mlp_probe(result, test["c_t"]))
        for metric, value in state_metrics(predictions, test["target"]).items():
            rows[f"c_t_state_{tag}_{metric}"] = value
        for m_value in M_TEST:
            subset = test["m"] == m_value
            if np.any(subset):
                for metric, value in state_metrics(predictions[subset], test["target"][subset]).items():
                    rows[f"c_t_state_{tag}_m{m_value}_{metric}"] = value

    digit_train = digit_features(apply, params, dataset, 0, config.readout_pairs, config.seed + 20)
    digit_val = digit_features(apply, params, dataset, 1, config.readout_pairs // 2, config.seed + 21)
    digit_test = digit_features(apply, params, dataset, 2, config.readout_pairs // 2, config.seed + 22)
    result = fit_digit_probe(
        keys[1],
        digit_train[0],
        digit_train[1],
        digit_val[0],
        digit_val[1],
        num_steps=config.readout_steps,
        batch_size=config.readout_batch_size,
        learning_rate=config.readout_lr,
        seed=config.seed,
    )
    rows["c_t_digit_accuracy"] = digit_accuracy(
        np.asarray(apply_mlp_probe(result, digit_test[0])), digit_test[1]
    )

    bounce_train = bounce_features(apply, params, dataset, 0, config.readout_pairs, config.seed + 30)
    bounce_val = bounce_features(apply, params, dataset, 1, config.readout_pairs // 2, config.seed + 31)
    bounce_test = bounce_features(apply, params, dataset, 2, config.readout_pairs // 2, config.seed + 32)
    result = fit_bounce_probe(
        keys[2],
        bounce_train[0],
        bounce_train[1],
        bounce_val[0],
        bounce_val[1],
        num_steps=config.readout_steps,
        batch_size=config.readout_batch_size,
        learning_rate=config.readout_lr,
        seed=config.seed,
    )
    for metric, value in bounce_metrics(
        np.asarray(apply_mlp_probe(result, bounce_test[0])), bounce_test[1]
    ).items():
        rows[f"c_t_bounce_{metric}"] = value

    features = test_by_query["final"]["c_t"][:, :EMBED_DIM]
    rows["c_t_variance"] = feature_variance(features)
    rows["c_t_cosine"] = mean_pairwise_cosine(features)
    rows["c_t_rank"] = effective_rank(features)
    return rows


def apply_mlp_probe(result, features: np.ndarray) -> np.ndarray:
    return np.asarray(apply_mlp(result.params, jnp.asarray(features)))


def run(objective: str, config: Config) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(config.seed, config.num_samples, mnist_dir=MNIST_DIR)
    params, apply, history = _train_model(config, objective, dataset, jax.random.key(config.seed))
    save_params(params, config.checkpoint_dir / "params.pkl")
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
