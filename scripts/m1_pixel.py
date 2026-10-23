"""M1: pixel-only baseline plus the common frozen-feature state readout."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from toy_jepa import constants
from toy_jepa.core import types
from toy_jepa.evaluation import protocol as experiment, readout
from toy_jepa.data.dataset import generate_dataset, pair_indices
from toy_jepa.evaluation.metrics import feature_variance
from toy_jepa.models import baseline
from toy_jepa.core.optim import adamw_init
from toy_jepa.core.train import train_step
from _artifacts import write_run_artifacts

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m1_pixel"
TITLE = "M1 pixel-only baseline + common state readout"
FEATURE_BATCH_SIZE = 256
apply_baseline_jit = jax.jit(baseline.apply_baseline)


class Config(NamedTuple):
    num_samples: int = 10_000
    num_steps: int = 3_000
    batch_size: int = 64
    k: int = experiment.K
    eval_every: int = 250
    eval_pairs: int = 1_000
    probe_pairs: int = 1_000
    readout_pairs: tuple[int, int, int] = constants.READOUT_PAIRS
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    embed_dim: int = 128
    hidden_dim: int = 256
    seed: int = constants.SEED
    run_dir: Path = RUN_DIR


def extract_features(
    params: baseline.BaselineParams, context: jax.Array
) -> tuple[types.Float32Array, types.Float32Array]:
    """Return frozen NumPy ``(c_t, z_hat)`` arrays in bounded chunks."""
    chunks = [
        apply_baseline_jit(params, context[i : i + FEATURE_BATCH_SIZE])
        for i in range(0, len(context), FEATURE_BATCH_SIZE)
    ]
    return (
        np.concatenate([np.asarray(c_t) for _, c_t, _ in chunks]),
        np.concatenate([np.asarray(z_hat) for _, _, z_hat in chunks]),
    )


def evaluate_pixels(
    params: baseline.BaselineParams,
    context: jax.Array,
    target_pixels: jax.Array,
) -> tuple[float, float]:
    """Return pixel MSE and the persistence reference."""
    pixels, _, _ = apply_baseline_jit(params, context)
    pixel_mse = jnp.mean((pixels - target_pixels) ** 2)
    persistence = jnp.mean((context[:, 1] - target_pixels) ** 2)
    return float(pixel_mse), float(persistence)


def train_baseline(
    config: Config,
    dataset: types.MovingMNIST,
    train_sequence: types.Int32Array,
    train_time: types.Int32Array,
    val_context: jax.Array,
    val_pixels: jax.Array,
    probe_context: jax.Array,
    key: jax.Array,
) -> tuple[types.TrainState[baseline.BaselineParams], list[experiment.EvalRow]]:
    """Train the pixel baseline and log loss/collapse diagnostics."""
    params = baseline.init_baseline(
        key, embed_dim=config.embed_dim, hidden_dim=config.hidden_dim
    )
    state = types.TrainState(params=params, opt_state=adamw_init(params))
    rng = np.random.default_rng(config.seed + 1)

    @jax.jit
    def step(state: types.TrainState[baseline.BaselineParams], batch):
        return train_step(
            state,
            batch,
            baseline.baseline_pixel_loss,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    history: list[experiment.EvalRow] = []
    for i in range(config.num_steps):
        batch = experiment.sample_batch(
            rng,
            dataset,
            train_sequence,
            train_time,
            batch_size=config.batch_size,
            k=config.k,
        )
        state, loss = step(state, (batch[0], batch[1]))
        if i % config.eval_every == 0 or i == config.num_steps - 1:
            val_mse, persistence = evaluate_pixels(
                state.params, val_context, val_pixels
            )
            _, probe_c, _ = apply_baseline_jit(state.params, probe_context)
            collapse = experiment.collapse_metrics(np.asarray(probe_c))
            row = experiment.EvalRow(
                step=i,
                train_loss=float(loss),
                val_loss=val_mse,
                persistence=persistence,
                variance=collapse["feature_variance"],
                cosine=collapse["mean_pairwise_cosine"],
                rank=collapse["effective_rank"],
            )
            history.append(row)
            print(
                f"step {i:4d}  train {row.train_loss:.6f}  "
                f"val {row.val_loss:.6f}  persistence {row.persistence:.6f}  "
                f"rank {row.rank:.2f}  var {row.variance:.6f}  "
                f"cos {row.cosine:.3f}"
            )
    return state, history


def run(config: Config) -> None:
    """Train the baseline, then train and evaluate the common readout."""
    dataset = generate_dataset(config.seed, config.num_samples)
    train_sequence, train_time = pair_indices(dataset, 0, config.k)
    val_sequence, val_time = pair_indices(dataset, 1, config.k)
    test_sequence, test_time = pair_indices(dataset, 2, config.k)
    val_context, val_pixels, _ = experiment.fixed_batch(
        dataset, val_sequence, val_time, num_pairs=config.eval_pairs, k=config.k
    )
    test_context, test_pixels, _ = experiment.fixed_batch(
        dataset, test_sequence, test_time, num_pairs=config.eval_pairs, k=config.k
    )
    probe_context, _, _ = experiment.fixed_batch(
        dataset, val_sequence, val_time, num_pairs=config.probe_pairs, k=config.k
    )
    print(f"frames: {dataset.frames.shape}")
    print(
        f"pairs: train={len(train_sequence):,} val={len(val_sequence):,} "
        f"test={len(test_sequence):,}"
    )

    model_key, c_key, z_key = jax.random.split(jax.random.key(config.seed), 3)
    state, history = train_baseline(
        config,
        dataset,
        train_sequence,
        train_time,
        val_context,
        val_pixels,
        probe_context,
        model_key,
    )
    test_mse, test_persistence = evaluate_pixels(
        state.params, test_context, test_pixels
    )

    readout_rng = np.random.default_rng(config.seed + 2)
    readout_arrays = [
        experiment.readout_split(dataset, readout_rng, split, n, k=config.k)
        for split, n in enumerate(config.readout_pairs)
    ]
    (
        (train_context, train_states),
        (val_context, val_states),
        (test_context, test_states),
    ) = readout_arrays
    train_c, train_z = extract_features(state.params, train_context)
    val_c, val_z = extract_features(state.params, val_context)
    test_c, test_z = extract_features(state.params, test_context)

    c_test, c_val = readout.fit_readout(
        c_key,
        train_c,
        train_states,
        val_c,
        val_states,
        test_c,
        test_states,
        seed=config.seed + 3,
    )
    z_test, z_val = readout.fit_readout(
        z_key,
        train_z,
        train_states,
        val_z,
        val_states,
        test_z,
        test_states,
        seed=config.seed + 4,
    )
    c_collapse = experiment.collapse_metrics(test_c)
    z_collapse = experiment.collapse_metrics(test_z)
    metrics: dict[str, float | int] = {
        "test_pixel_mse": test_mse,
        "test_persistence_mse": test_persistence,
        "val_pixel_mse": history[-1].val_loss,
        "val_persistence_mse": history[-1].persistence,
        "target_variance": feature_variance(np.asarray(test_states)),
        "num_samples": config.num_samples,
        "num_steps": config.num_steps,
        "k": config.k,
    }
    metrics.update(
        experiment.summarize_readouts(
            {
                "c_t": (c_test, c_val, c_collapse),
                "z_hat": (z_test, z_val, z_collapse),
            }
        )
    )
    write_run_artifacts(
        config, TITLE, history, metrics, {"c_t": c_test, "z_hat": z_test}
    )

    print(f"test pixel MSE: {test_mse:.6f}")
    print(f"test persistence MSE: {test_persistence:.6f}")
    for name, result in (("c_t", c_test), ("z_hat", z_test)):
        print(
            f"{name} readout: MSE={result['state_mse']:.6f}  "
            f"pos MAE={result['position_mae']:.3f} px  "
            f"vel MAE={result['velocity_mae']:.3f} px/frame  "
            f"R²={result['r2']:.3f}"
        )
    print(f"wrote {config.run_dir}")


def main() -> None:
    run(Config())


if __name__ == "__main__":
    main()
