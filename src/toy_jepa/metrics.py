"""Evaluation metrics and simple linear/kNN probes (NumPy)."""

from __future__ import annotations

import numpy as np

from .constants import POSITION_SCALE, VELOCITY_SCALE
from .types import Float32Array, Int8Array


def _as_float64(x: Float32Array) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def effective_rank(features: Float32Array) -> float:
    """Participation ratio of the centered feature spectrum."""
    x = _as_float64(features)
    x = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, compute_uv=False)
    energy = np.sum(singular_values**2)
    if energy <= 0.0:
        return 0.0
    return float(np.sum(singular_values) ** 2 / energy)


def feature_variance(features: Float32Array) -> float:
    """Mean per-dimension variance."""
    x = _as_float64(features)
    return float(np.mean(np.var(x, axis=0)))


def mean_pairwise_cosine(features: Float32Array) -> float:
    """Mean cosine similarity over unordered pairs of feature rows."""
    x = _as_float64(features)
    if len(x) < 2:
        return 0.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    similarities = x @ x.T
    mask = ~np.eye(len(x), dtype=bool)
    return float(similarities[mask].mean())


def state_readout_metrics(
    predictions: Float32Array,
    targets: Float32Array,
    *,
    position_scale: float = POSITION_SCALE,
    velocity_scale: float = VELOCITY_SCALE,
) -> dict[str, float]:
    """Metrics for an 8-d state readout.

    Targets use the shared normalization (position / 36, velocity / 3), so
    ``state_mse`` is in normalized units while the MAEs are converted back to
    pixels and pixels/frame. The layout is four positions followed by four
    velocities: ``(x, y)`` for each of the two digits.
    """
    prediction = _as_float64(predictions)
    target = _as_float64(targets)
    error = prediction - target

    residual = np.sum(error**2, axis=0)
    total = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)

    return {
        "state_mse": float(np.mean(error**2)),
        "position_mae": float(np.mean(np.abs(error[:, :4])) * position_scale),
        "velocity_mae": float(np.mean(np.abs(error[:, 4:])) * velocity_scale),
        "r2": float(np.mean(r2)),
    }


def ridge_r2(
    train_features: Float32Array,
    train_targets: Float32Array,
    test_features: Float32Array,
    test_targets: Float32Array,
    *,
    alpha: float = 1e-3,
) -> float:
    """Mean per-dimension R² of a closed-form ridge regression."""
    x_train = _as_float64(train_features)
    y_train = _as_float64(train_targets)
    x_test = _as_float64(test_features)
    y_test = _as_float64(test_targets)

    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_center = x_train - x_mean
    y_center = y_train - y_mean

    num_train, dim = x_center.shape
    if dim <= num_train:
        gram = x_center.T @ x_center + alpha * np.eye(dim)
        weights = np.linalg.solve(gram, x_center.T @ y_center)
        predictions = (x_test - x_mean) @ weights + y_mean
    else:
        # Dual form avoids solving a d x d system when d >> n.
        kernel = x_center @ x_center.T + alpha * np.eye(num_train)
        dual = np.linalg.solve(kernel, y_center)
        predictions = (x_test - x_mean) @ x_center.T @ dual + y_mean
    residual = np.sum((y_test - predictions) ** 2, axis=0)
    total = np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    return float(np.mean(r2))


def knn_accuracy(
    train_features: Float32Array,
    train_labels: Int8Array,
    test_features: Float32Array,
    test_labels: Int8Array,
    *,
    k: int = 5,
) -> float:
    """Mean kNN accuracy over the label columns, using cosine similarity."""
    x_train = _as_float64(train_features)
    x_test = _as_float64(test_features)
    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)

    x_train = x_train / np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True), 1e-12)
    x_test = x_test / np.maximum(np.linalg.norm(x_test, axis=1, keepdims=True), 1e-12)

    k = min(k, len(x_train))
    neighbors = np.argsort(-(x_test @ x_train.T), axis=1)[:, :k]

    accuracies = []
    for column in range(y_train.shape[1]):
        votes = y_train[neighbors, column]
        predictions = np.array(
            [
                np.bincount(row, minlength=10).argmax()
                for row in votes
            ],
            dtype=np.int8,
        )
        accuracies.append(np.mean(predictions == y_test[:, column]))
    return float(np.mean(accuracies))
