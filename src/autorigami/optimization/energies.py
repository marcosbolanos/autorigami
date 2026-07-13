from __future__ import annotations

import numpy as np

from autorigami._native import segment_segment_distance
from autorigami.geometry.curvature import get_polyline_angles
from autorigami.geometry.separation import get_candidate_intersecting_edges
from autorigami.types import EdgeIndex, Polyline


def sparse_separation_energy(
    polyline: Polyline,
    min_distance: float,
    candidate_pairs: list[tuple[EdgeIndex, EdgeIndex]] | None = None,
    n_ignored_adjacent_edges: int = 1,
) -> float:
    if candidate_pairs is None:
        candidate_pairs = get_candidate_intersecting_edges(
            polyline,
            min_distance=min_distance,
            n_ignored_adjacent_edges=n_ignored_adjacent_edges,
        )

    distances = segment_segment_distance(polyline, candidate_pairs)
    violations = np.array(
        [max(0.0, min_distance - distance) for distance, _, _ in distances],
        dtype=np.float32,
    )
    return float(0.5 * np.sum(violations * violations))


def curvature_violation_energy(polyline: Polyline, target_angle: float) -> float:
    angles = get_polyline_angles(polyline)
    violations = np.maximum(0.0, angles - np.float32(target_angle))
    return float(0.5 * np.sum(violations * violations))
