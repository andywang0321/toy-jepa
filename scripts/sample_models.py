"""Compare target, Pixel-MSE, Pixel-FM, and JEPA-decoded futures.

Rows are horizons; columns are target/model. JEPA has no native pixel head, so
this script trains a small flow head on frozen JEPA c_t features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from tiny_worlds.friction_mnist import generate_dataset

from toy_jepa.checkpoint import load_params, save_params
from toy_jepa.config import M_TRAIN, FINAL_PROB, K_TRAIN, MNIST_DIR
from toy_jepa.data import WindowBatch, make_batch, sample_batch
from toy_jepa.mlp import apply_mlp, init_mlp
from toy_jepa.models import (
    FlowParams,
    JEPAParams,
    PixelParams,
    apply_context,
    apply_pixel,
    sample_flow,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = int(os.environ.get("P2_SEED", "0"))
SAMPLE_INDEX = int(os.environ.get("SAMPLE_INDEX", "0"))
M = 8
K_VALUES = (3, 12, 24, 48)
FLOW_STEPS = int(os.environ.get("FLOW_STEPS", "24"))
HEAD_STEPS = int(os.environ.get("HEAD_STEPS", "1500"))
HEAD_SAMPLES = int(os.environ.get("HEAD_SAMPLES", "3000"))
HEAD_BATCH = int(os.environ.get("HEAD_BATCH", "64"))


def _load(path: Path) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, load_params(path))


def _target_batch(dataset, sequence: int, time: int, k: int, final: bool) -> WindowBatch:
    return make_batch(
        dataset,
        np.asarray([sequence], dtype=np.int32),
        np.asarray([time], dtype=np.int32),
        np.asarray([M], dtype=np.int32),
        np.asarray([k], dtype=np.int32),
        np.asarray([final]),
    )


def _head_loss(head: Any, batch: tuple[jax.Array, ...], key: jax.Array) -> jax.Array:
    c_t, query, target = batch
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, target.shape)
    time = jax.random.uniform(time_key, (target.shape[0], 1))
    velocity = apply_mlp(
        head,
        jnp.concatenate([c_t, query, time], axis=-1),
    ).reshape(-1, 64, 64)
    return jnp.mean((velocity - (target - noise)) ** 2)


def _train_head(jepa: JEPAParams, dataset, key: jax.Array) -> Any:
    rng = np.random.default_rng(SEED + 31)
    batch = sample_batch(
        rng,
        dataset,
        0,
        HEAD_SAMPLES,
        m_values=M_TRAIN,
        k_values=K_TRAIN,
        final_prob=FINAL_PROB,
    )
    c_t, query = jax.jit(lambda b: apply_context(jepa.context, b))(batch)
    data = (c_t, query, batch.target_frame)
    head = init_mlp(key, [2 * c_t.shape[-1] + 1, 512, 64 * 64])
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    state = optimizer.init(head)

    @jax.jit
    def step(head, state, batch, step_key):
        loss, grads = jax.value_and_grad(_head_loss)(head, batch, step_key)
        updates, state = optimizer.update(grads, state, head)
        return optax.apply_updates(head, updates), state, loss

    train_key = jax.random.key(SEED + 41)
    for step_number in range(HEAD_STEPS):
        rows = rng.integers(0, HEAD_SAMPLES, HEAD_BATCH)
        train_key, step_key = jax.random.split(train_key)
        head, state, loss = step(
            head, state, tuple(value[rows] for value in data), step_key
        )
        if step_number % 250 == 0:
            print(f"jepa flow head {step_number:4d} loss {float(loss):.4f}")
    return head


def _sample_head(head: Any, c_t: jax.Array, query: jax.Array, key: jax.Array) -> jax.Array:
    frame = jax.random.normal(key, (c_t.shape[0], 64, 64))
    for index in range(FLOW_STEPS):
        time = jnp.full((c_t.shape[0], 1), index / FLOW_STEPS)
        velocity = apply_mlp(
            head,
            jnp.concatenate([c_t, query, time], axis=-1),
        ).reshape(-1, 64, 64)
        frame = frame + velocity / FLOW_STEPS
    return jnp.clip(frame, 0.0, 1.0)


def main() -> None:
    dataset = generate_dataset(SEED, 100, mnist_dir=MNIST_DIR)
    max_k = max(K_VALUES)
    candidates = np.flatnonzero(
        (dataset.split == 2) & (dataset.stop_time >= M - 1 + max_k)
    )
    if len(candidates) == 0:
        raise SystemExit("no sequence has enough room for the requested horizons.")
    sequence = int(candidates[SAMPLE_INDEX % len(candidates)])
    time = M - 1
    stop = int(dataset.stop_time[sequence])

    pixel = cast(PixelParams, _load(ROOT / "checkpoints/p2/pixel" / f"seed_{SEED}" / "params.pkl"))
    pixel_fm = cast(FlowParams, _load(ROOT / "checkpoints/p2/pixel_fm" / f"seed_{SEED}" / "params.pkl"))
    jepa = cast(JEPAParams, _load(ROOT / "checkpoints/p2/jepa" / f"seed_{SEED}" / "params.pkl"))

    head = _train_head(jepa, dataset, jax.random.key(SEED + 71))
    save_params(head, ROOT / "checkpoints/p2/jepa" / f"seed_{SEED}" / "flow_head.pkl")

    queries = [(k, False) for k in K_VALUES] + [(stop - time, True)]
    rows = []
    for index, (k, final) in enumerate(queries):
        batch = _target_batch(dataset, sequence, time, k, final)
        target = np.asarray(batch.target_frame[0])
        pixel_frame = np.asarray(apply_pixel(pixel, batch)[2][0])
        flow_frame = np.asarray(
            sample_flow(
                pixel_fm,
                batch,
                jax.random.key(SEED + index),
                steps=FLOW_STEPS,
            )[0]
        )
        c_t, query = apply_context(jepa.context, batch)
        jepa_frame = np.asarray(
            _sample_head(
                head,
                c_t,
                query,
                jax.random.key(SEED + 100 + index),
            )[0]
        )
        rows.append((k, final, target, pixel_frame, flow_frame, jepa_frame))

    columns = ("target", "pixel-MSE", "pixel-FM", "JEPA + flow")
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(8, 2.0 * len(rows)))
    for row, (k, final, target, pixel_frame, flow_frame, jepa_frame) in enumerate(rows):
        label = "FINAL" if final else f"k={k}"
        for column, image in enumerate((target, pixel_frame, flow_frame, jepa_frame)):
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_ylabel(label if column == 0 else "")
            if row == 0:
                axis.set_title(columns[column])
    fig.suptitle(
        f"friction world sample {sequence}, m={M}, context t={time}, stop={stop}",
        fontsize=10,
    )
    fig.tight_layout()
    out = ROOT / "runs/analysis/samples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
