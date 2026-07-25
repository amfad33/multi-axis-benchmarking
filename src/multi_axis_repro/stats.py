from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


def rdm_vector(features: np.ndarray, metric: str = "correlation") -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    return np.nan_to_num(pdist(features, metric=metric), copy=False)


def upper_triangle(square: np.ndarray) -> np.ndarray:
    square = np.asarray(square)
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError(f"RDM must be square, got {square.shape}")
    return square[np.triu_indices(square.shape[0], 1)]


def square_rdm(vector: np.ndarray) -> np.ndarray:
    count = len(vector)
    n = (1 + math.sqrt(1 + 8 * count)) / 2
    if not n.is_integer():
        raise ValueError(f"RDM length {count} is not triangular")
    return squareform(vector, checks=False)


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values), method="average").astype(float)
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    if norm == 0:
        raise ValueError("Cannot rank-standardize a constant vector")
    return ranks / norm


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if np.any((values < 0) | (values > 1) | ~np.isfinite(values)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()
