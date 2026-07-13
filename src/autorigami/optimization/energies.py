from __future__ import annotations

import numpy as np

from autorigami.geometry.curvature import get_polyline_angles
from autorigami.types import Polyline

_EPSILON = np.float32(1e-12)


def curvature_violation_energy(polyline: Polyline, target_angle: float) -> float:
    angles = get_polyline_angles(polyline)
    violations = np.maximum(0.0, angles - np.float32(target_angle))
    return float(0.5 * np.sum(violations * violations))


def curvature_violation_energy_gradient(
    polyline: Polyline,
    target_angle: float,
) -> Polyline:
    """Return the curvature violation energy gradient for every vertex.

    Each inner angle contributes to its preceding, center, and following vertex.
    The three contribution arrays are accumulated by slices so adjacent angle
    stencils are handled without a Python loop.
    """
    edge_vectors = polyline[1:] - polyline[:-1]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    edge_directions = edge_vectors / edge_lengths

    left_directions = edge_directions[:-1]
    right_directions = edge_directions[1:]
    cosines = np.sum(left_directions * right_directions, axis=1, keepdims=True)
    cosines = np.clip(cosines, -1.0, 1.0)
    angles = np.arccos(cosines)
    violations = np.maximum(
        np.float32(0.0),
        angles - np.float32(target_angle),
    )

    sines = np.sqrt(np.maximum(np.float32(0.0), np.float32(1.0) - cosines**2))
    angle_gradient_left_edge = -(
        right_directions - cosines * left_directions
    ) / np.maximum(edge_lengths[:-1] * sines, _EPSILON)
    angle_gradient_right_edge = -(
        left_directions - cosines * right_directions
    ) / np.maximum(edge_lengths[1:] * sines, _EPSILON)

    weighted_left = violations * angle_gradient_left_edge
    weighted_right = violations * angle_gradient_right_edge
    gradient = np.zeros_like(polyline, dtype=np.float32)
    gradient[:-2] -= weighted_left
    gradient[1:-1] += weighted_left - weighted_right
    gradient[2:] += weighted_right
    return gradient
