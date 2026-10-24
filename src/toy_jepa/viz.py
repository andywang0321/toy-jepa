"""Plotting helpers (impure)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluation.protocol import EvalRow


def plot_dashboard(
    history: list[EvalRow],
    readout_metrics: Mapping[str, Mapping[str, float]],
    path: Path,
    *,
    title: str = "Collapse dashboard",
) -> None:
    """Plot loss/persistence, collapse diagnostics, and readout MSE bars.

    ``history`` is shared by every arm. ``readout_metrics`` maps a feature name
    (for example ``"c_t"`` or ``"z_hat"``) to the metrics returned by
    :func:`toy_jepa.evaluation.metrics.state_readout_metrics`; the final panel compares
    their ``state_mse``.
    """
    if not history:
        raise ValueError("dashboard needs at least one evaluation row.")

    steps = [row.step for row in history]
    train_losses = [row.train_loss for row in history]
    val_losses = [row.val_loss for row in history]
    persistence = [row.persistence for row in history]
    variances = [max(row.variance, 1e-12) for row in history]
    cosines = [row.cosine for row in history]
    ranks = [row.rank for row in history]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))

    axes[0, 0].plot(steps, train_losses, marker="o", markersize=3, label="train")
    axes[0, 0].plot(steps, val_losses, marker="o", markersize=3, label="val")
    axes[0, 0].plot(steps, persistence, linestyle="--", label="persistence")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()

    axes[0, 1].plot(steps, variances, marker="o", markersize=3, color="tab:green")
    axes[0, 1].set_ylabel("feature variance")
    axes[0, 1].set_yscale("log")

    axes[0, 2].plot(steps, cosines, marker="o", markersize=3, color="tab:red")
    axes[0, 2].set_ylabel("mean pairwise cosine")
    axes[0, 2].set_ylim(0.0, 1.05)

    axes[1, 0].plot(steps, ranks, marker="o", markersize=3, color="tab:orange")
    axes[1, 0].set_ylabel("effective rank")
    axes[1, 0].set_ylim(bottom=0.0)

    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)

    axes[1, 1].remove()
    axes[1, 2].remove()
    readout_axis = fig.add_subplot(2, 3, (5, 6))
    names = list(readout_metrics)
    if names:
        values = [float(readout_metrics[name].get("state_mse", np.nan)) for name in names]
        bars = readout_axis.bar(np.arange(len(names)), values)
        readout_axis.set_xticks(np.arange(len(names)), names)
        readout_axis.set_ylabel("state MSE (normalized)")
        readout_axis.set_ylim(bottom=0.0)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                readout_axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    else:
        readout_axis.text(
            0.5,
            0.5,
            "no readout metrics",
            ha="center",
            va="center",
            transform=readout_axis.transAxes,
        )
        readout_axis.set_xticks([])
        readout_axis.set_yticks([])
    readout_axis.set_title("post-hoc readout", fontsize=10)
    readout_axis.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
