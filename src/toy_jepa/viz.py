"""Plotting helpers (impure)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .constants import DIGIT_SIZE
from .experiment import EvalRow
from .types import Float32Array, Int8Array, UInt8Array

_COLORS = ("tab:red", "tab:cyan")


def _metadata_line(
    position: Float32Array,
    velocity: Float32Array,
    digit_label: Int8Array | None,
    frame: int,
    digit: int,
) -> str:
    """Format one digit's metadata for the footer."""
    x, y = position[frame, digit]
    vx, vy = velocity[frame, digit]
    label = "" if digit_label is None else f"label={int(digit_label[digit])}  "
    return (
        f"d{digit}  {label}"
        f"x={x:5.1f}  y={y:5.1f}  vx={vx:+5.2f}  vy={vy:+5.2f}"
    )


def visualize_sample(
    sample: UInt8Array,
    position: Float32Array,
    velocity: Float32Array,
    digit_label: Int8Array | None = None,
    *,
    fps: int = 15,
    velocity_scale: float = 5.0,
) -> None:
    """Play a (T, H, W) video with per-digit position/velocity metadata.

    The arrow at each digit's center is the velocity vector, scaled by
    ``velocity_scale`` for visibility. The footer prints the top-left position
    and velocity in pixels and pixels/frame.
    """
    num_frames = sample.shape[0]
    height, width = sample.shape[1:]
    centers = position + (DIGIT_SIZE / 2.0)

    with plt.ion():
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(bottom=0.18)
        image = ax.imshow(sample[0], cmap="gray", vmin=0, vmax=255)
        arrows = ax.quiver(
            centers[0, :, 0],
            centers[0, :, 1],
            velocity[0, :, 0] * velocity_scale,
            velocity[0, :, 1] * velocity_scale,
            color=_COLORS,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="tail",
            width=0.008,
        )
        footer_y = (0.10, 0.02)
        footers = [
            fig.text(
                0.01,
                footer_y[digit],
                "",
                color=_COLORS[digit],
                family="monospace",
                fontsize=8,
                va="bottom",
            )
            for digit in range(2)
        ]

        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "pos = top-left (px), vel = px/frame; "
            f"arrows = velocity x{velocity_scale:g}",
            fontsize=8,
        )

        for frame in range(num_frames):
            image.set_data(sample[frame])
            arrows.set_offsets(centers[frame])
            arrows.set_UVC(
                velocity[frame, :, 0] * velocity_scale,
                velocity[frame, :, 1] * velocity_scale,
            )
            for digit, footer in enumerate(footers):
                footer.set_text(
                    _metadata_line(position, velocity, digit_label, frame, digit)
                )
            plt.draw()
            plt.pause(1.0 / fps)
    plt.show()


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
    :func:`toy_jepa.metrics.state_readout_metrics`; the final panel compares
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
    axes[0, 0].plot(
        steps, persistence, linestyle="--", label="persistence"
    )
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
        values = [
            float(readout_metrics[name].get("state_mse", np.nan))
            for name in names
        ]
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
            0.5, 0.5, "no readout metrics", ha="center", va="center",
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

