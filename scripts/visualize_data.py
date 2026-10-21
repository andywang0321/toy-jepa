"""Visualize a few generated Moving MNIST sequences.

Generates ``VISUALIZE_SAMPLES`` sequences in memory and plays them with
velocity arrows and per-digit position/velocity metadata. All defaults live in
``constants.py``; the script takes no command-line arguments.

Usage:
    uv run python scripts/visualize_data.py
"""

from __future__ import annotations

from toy_jepa.constants import SEED, VISUALIZE_SAMPLES
from toy_jepa.data import generate_dataset
from toy_jepa.viz import visualize_sample


def main() -> None:
    """Play a few samples."""
    dataset = generate_dataset(SEED, VISUALIZE_SAMPLES)
    for frames, position, velocity, digit_label in zip(
        dataset.frames,
        dataset.position,
        dataset.velocity,
        dataset.digit_label,
    ):
        visualize_sample(frames, position, velocity, digit_label)


if __name__ == "__main__":
    main()
