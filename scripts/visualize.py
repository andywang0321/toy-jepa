"""Visualize a few friction Moving MNIST sequences."""

from __future__ import annotations

from tiny_worlds.friction_mnist import generate_dataset, visualize_sample
from toy_jepa.config import MNIST_DIR


def main() -> None:
    dataset = generate_dataset(0, 3, mnist_dir=MNIST_DIR)
    for index in range(3):
        visualize_sample(dataset, index)


if __name__ == "__main__":
    main()
