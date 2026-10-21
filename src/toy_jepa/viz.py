"""Plotting helpers (impure)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .constants import DIGIT_SIZE
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
