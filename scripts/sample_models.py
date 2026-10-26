"""Compare target, Pixel-MSE, Pixel-FM, JEPA-flow, and JEPA-VAE futures.

Rows are models; columns are horizons. JEPA has no native pixel head, so this
script trains flow and VAE heads on frozen JEPA c_t features.
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
from tiny_worlds.friction_mnist import generate_dataset

from toy_jepa.checkpoint import load_params, save_params
from toy_jepa.config import FINAL_PROB, K_TRAIN, MNIST_DIR, M_TRAIN
from toy_jepa.data import WindowBatch, make_batch, sample_batch
from toy_jepa.jepa_heads import (
    sample_flow_head,
    sample_vae_head,
    train_flow_head,
    train_vae_head,
)
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
    flow_head = train_flow_head(
        *data,
        key=jax.random.key(SEED + 71),
        steps=HEAD_STEPS,
        batch_size=HEAD_BATCH,
    )
    vae_head = train_vae_head(
        *data,
        key=jax.random.key(SEED + 81),
        steps=HEAD_STEPS,
        batch_size=HEAD_BATCH,
    )
    save_params(flow_head, ROOT / "checkpoints/p2/jepa" / f"seed_{SEED}" / "flow_head.pkl")
    save_params(vae_head, ROOT / "checkpoints/p2/jepa" / f"seed_{SEED}" / "vae_head.pkl")

    queries = [(k, False) for k in K_VALUES] + [(stop - time, True)]
    rows: dict[str, list[np.ndarray]] = {name: [] for name in ("target", "pixel-MSE", "pixel-FM", "JEPA flow", "JEPA VAE")}
    for index, (k, final) in enumerate(queries):
        batch = _target_batch(dataset, sequence, time, k, final)
        c_t, query = apply_context(jepa.context, batch)
        rows["target"].append(np.asarray(batch.target_frame[0]))
        rows["pixel-MSE"].append(np.asarray(apply_pixel(pixel, batch)[2][0]))
        rows["pixel-FM"].append(
            np.asarray(sample_flow(pixel_fm, batch, jax.random.key(SEED + index), steps=FLOW_STEPS)[0])
        )
        rows["JEPA flow"].append(
            np.asarray(sample_flow_head(flow_head, c_t, query, jax.random.key(SEED + 100 + index), steps=FLOW_STEPS)[0])
        )
        rows["JEPA VAE"].append(
            np.asarray(sample_vae_head(vae_head, c_t, query, jax.random.key(SEED + 200 + index))[0])
        )

    labels = ["FINAL" if final else f"k={k}" for k, final in queries]
    fig, axes = plt.subplots(len(rows), len(queries), figsize=(2.0 * len(queries), 2.0 * len(rows)))
    for row, (name, images) in enumerate(rows.items()):
        for column, image in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(name, fontsize=9)
            if row == 0:
                axis.set_title(labels[column], fontsize=9)
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
