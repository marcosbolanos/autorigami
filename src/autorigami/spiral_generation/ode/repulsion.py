from __future__ import annotations

import numpy as np


def repulsive_potential_gradient(
    point: np.ndarray,
    history: np.ndarray,
    range_world: float,
) -> np.ndarray:
    """Gradient of R(x)=sum exp(-(r/delta)^2) over historical points."""
    if history.size == 0:
        return np.zeros(3, dtype=float)

    delta = max(range_world, 1e-8)
    diff = point[None, :] - history
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    weights = np.exp(-dist_sq / (delta * delta))

    grad = (-2.0 / (delta * delta)) * np.sum(weights[:, None] * diff, axis=0)
    return grad
