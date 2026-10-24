"""M5: EMA JEPA plus SIGReg on the context feature."""

from __future__ import annotations

from functools import partial
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
from toy_jepa.models import cnn, jepa_sigreg
from toy_jepa.core.optim import adamw_init
from toy_jepa.core.train import ema_train_step
from _artifacts import write_run_artifacts

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "m5_sigreg"
TITLE = "M5 EMA + SIGReg JEPA + common state readout"
FEATURE_BATCH_SIZE = 256
SIGREG_WEIGHT = 0.03
apply_jepa_sigreg_jit = jax.jit(jepa_sigreg.apply_jepa_sigreg)
apply_cnn_jit = jax.jit(cnn.apply_cnn)


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
    ema_decay: float = 0.99
    sigreg_weight: float = SIGREG_WEIGHT
    run_dir: Path = RUN_DIR


def extract_features(
    params: jepa_sigreg.JEPASigRegParams, context: jax.Array
) -> tuple[types.Float32Array, types.Float32Array]:
    """Return frozen NumPy ``(c_t, z_hat)`` arrays in bounded chunks."""
    chunks = [
        apply_jepa_sigreg_jit(params, context[i : i + FEATURE_BATCH_SIZE])
        for i in range(0, len(context), FEATURE_BATCH_SIZE)
    ]
    return (
        np.concatenate([np.asarray(c_t) for _, c_t in chunks]),
        np.concatenate([np.asarray(z_hat) for z_hat, _ in chunks]),
    )


def loss_and_persistence(
    params: jepa_sigreg.JEPASigRegParams,
    target_encoder: types.CNNParams,
    context: jax.Array,
    target_frame: jax.Array,
) -> tuple[float, float]:
    """Return latent loss and the online-encoder persistence reference."""
    z_hat, _ = apply_jepa_sigreg_jit(params, context)
    z_target = apply_cnn_jit(target_encoder, target_frame[..., None])
    z_current = apply_cnn_jit(params.encoder, context[:, 1][..., None])
    z_future = apply_cnn_jit(params.encoder, target_frame[..., None])
    loss = jnp.mean((z_hat - z_target) ** 2)
    persistence = jnp.mean((z_current - z_future) ** 2)
    return float(loss), float(persistence)


def evaluate(
    params: jepa_sigreg.JEPASigRegParams,
    target_encoder: types.CNNParams,
    val_context: jax.Array,
    val_target: jax.Array,
    probe_context: jax.Array,
    *,
    step: int,
    train_loss: float,
) -> experiment.EvalRow:
    """Evaluate loss, persistence, and collapse diagnostics on fixed probes."""
    val_loss, persistence = loss_and_persistence(params, target_encoder, val_context, val_target)
    _, c_t = apply_jepa_sigreg_jit(params, probe_context)
    collapse = experiment.collapse_metrics(np.asarray(c_t))
    return experiment.EvalRow(
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        persistence=persistence,
        variance=collapse["feature_variance"],
        cosine=collapse["mean_pairwise_cosine"],
        rank=collapse["effective_rank"],
    )


def train(
    config: Config,
    dataset: types.MovingMNIST,
    train_sequence: types.Int32Array,
    train_time: types.Int32Array,
    val_context: jax.Array,
    val_target: jax.Array,
    probe_context: jax.Array,
    key: jax.Array,
) -> tuple[jepa_sigreg.JEPASigRegParams, types.CNNParams, list[experiment.EvalRow]]:
    """Train the model and log shared evaluation rows."""
    params = jepa_sigreg.init_jepa_sigreg(key, embed_dim=config.embed_dim, hidden_dim=config.hidden_dim)
    state = types.EMAState(
        train_state=types.TrainState(params=params, opt_state=adamw_init(params)),
        target_encoder=params.encoder,
    )
    rng = np.random.default_rng(config.seed + 1)
    loss_fn = partial(
        jepa_sigreg.jepa_sigreg_loss,
        key=jax.random.key(config.seed + 5),
        weight=config.sigreg_weight,
    )

    @jax.jit
    def step(state: types.EMAState[jepa_sigreg.JEPASigRegParams], batch):
        return ema_train_step(
            state,
            batch,
            loss_fn,
            lambda p: p.encoder,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            decay=config.ema_decay,
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
            row = evaluate(
                state.train_state.params,
                state.target_encoder,
                val_context,
                val_target,
                probe_context,
                step=i,
                train_loss=float(loss),
            )
            history.append(row)
            print(
                f"step {i:4d}  train {row.train_loss:.6f}  "
                f"val {row.val_loss:.6f}  persistence {row.persistence:.6f}  "
                f"rank {row.rank:.2f}  var {row.variance:.6f}  "
                f"cos {row.cosine:.3f}"
            )
    return state.train_state.params, state.target_encoder, history


def run(config: Config) -> None:
    dataset = generate_dataset(config.seed, config.num_samples)
    train_sequence, train_time = pair_indices(dataset, 0, config.k)
    val_sequence, val_time = pair_indices(dataset, 1, config.k)
    test_sequence, test_time = pair_indices(dataset, 2, config.k)
    val_context, val_target, _ = experiment.fixed_batch(
        dataset, val_sequence, val_time, num_pairs=config.eval_pairs, k=config.k
    )
    test_context, test_target, _ = experiment.fixed_batch(
        dataset, test_sequence, test_time, num_pairs=config.eval_pairs, k=config.k
    )
    probe_context, _, _ = experiment.fixed_batch(
        dataset, val_sequence, val_time, num_pairs=config.probe_pairs, k=config.k
    )
    print(f"frames: {dataset.frames.shape}")
    print(f"pairs: train={len(train_sequence):,} val={len(val_sequence):,} test={len(test_sequence):,}")

    model_key, c_key, z_key = jax.random.split(jax.random.key(config.seed), 3)
    params, target_encoder, history = train(
        config,
        dataset,
        train_sequence,
        train_time,
        val_context,
        val_target,
        probe_context,
        model_key,
    )
    test_loss, test_persistence = loss_and_persistence(params, target_encoder, test_context, test_target)
    test_targets = apply_cnn_jit(target_encoder, test_target[..., None])
    target_variance = feature_variance(np.asarray(test_targets))

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
    train_c, train_z = extract_features(params, train_context)
    val_c, val_z = extract_features(params, val_context)
    test_c, test_z = extract_features(params, test_context)

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
        "test_loss": test_loss,
        "test_latent_persistence_mse": test_persistence,
        "val_loss": history[-1].val_loss,
        "val_latent_persistence_mse": history[-1].persistence,
        "target_variance": target_variance,
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
    write_run_artifacts(config, TITLE, history, metrics, {"c_t": c_test, "z_hat": z_test})

    print(f"test latent MSE: {test_loss:.6f}  persistence: {test_persistence:.6f}")
    for name, result in (("c_t", c_test), ("z_hat", z_test)):
        print(
            f"{name} readout: MSE={result['state_mse']:.6f}  "
            f"pos MAE={result['position_mae']:.3f} px  "
            f"vel MAE={result['velocity_mae']:.3f} px/frame  "
            f"R²={result['r2']:.3f}"
        )
    print(
        f"c_t collapse: var={c_collapse['feature_variance']:.6f}  "
        f"cos={c_collapse['mean_pairwise_cosine']:.3f}  "
        f"rank={c_collapse['effective_rank']:.2f}"
    )
    print(f"wrote {config.run_dir}")


def main() -> None:
    run(Config())


if __name__ == "__main__":
    main()
