"""Visualize a few friction Moving MNIST sequences.

Generates ``VISUALIZE_SAMPLES`` sequences in memory and plays them with
velocity arrows and per-digit position/velocity metadata. All defaults live in
``constants.py``; the script takes no command-line arguments.

Usage:
    uv run python scripts/visualize_data.py
"""

from __future__ import annotations

from toy_jepa.constants import SEED, VISUALIZE_SAMPLES
from toy_jepa.data.dataset import generate_dataset
from tiny_worlds.friction_mnist import visualize_sample


def main() -> None:
    """Play a few samples."""
    dataset = generate_dataset(SEED, VISUALIZE_SAMPLES)
    for index in range(VISUALIZE_SAMPLES):
        visualize_sample(dataset, index)


if __name__ == "__main__":
    main()
