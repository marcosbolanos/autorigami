from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from autorigami.geometry.curvature import (
    DEFAULT_MAX_ANGLE_RADIANS,
    get_polyline_angles,
)
from autorigami.geometry.separation import (
    DEFAULT_MIN_DISTANCE,
    check_self_intersections,
)
from autorigami.types import Polyline


@dataclass(frozen=True)
class PolylineViolationMasks:
    """Per-edge and per-vertex masks accepted directly by renderers."""

    separation_edges: npt.NDArray[np.bool_]
    curvature_vertices: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class PolylineValidationResult:
    """Standard curvature and separation validation for one centerline."""

    maximum_angle: float
    curvature_violation_count: int
    separation_violation_count: int
    minimum_violating_distance: float | None
    ignored_adjacent_edges: int
    violation_masks: PolylineViolationMasks | None

    @property
    def valid(self) -> bool:
        return (
            self.curvature_violation_count == 0 and self.separation_violation_count == 0
        )


def validate_polyline(
    polyline: Polyline,
    *,
    maximum_angle: float = DEFAULT_MAX_ANGLE_RADIANS,
    minimum_distance: float = DEFAULT_MIN_DISTANCE,
    valid_curvature_ignored_adjacent_edges: int = 100,
    include_violation_masks: bool = False,
) -> PolylineValidationResult:
    """Validate both hard constraints, optionally retaining renderer masks.

    A long local-edge exclusion is sound only when curvature is valid. If any
    angle is invalid, separation falls back to excluding immediate neighbors
    only, so an invalid curve cannot hide collisions behind the curvature
    assumption.
    """
    assert polyline.ndim == 2 and polyline.shape[1] == 3
    assert len(polyline) >= 3
    assert 0.0 < maximum_angle < np.pi
    assert minimum_distance > 0.0
    assert valid_curvature_ignored_adjacent_edges >= 1

    angles = get_polyline_angles(polyline)
    curvature_inner_mask = angles > maximum_angle
    curvature_violation_count = int(np.count_nonzero(curvature_inner_mask))
    ignored_adjacent_edges = (
        valid_curvature_ignored_adjacent_edges if curvature_violation_count == 0 else 1
    )
    separation = check_self_intersections(
        polyline,
        min_euclid_distance=minimum_distance,
        n_ignored_adjacent_edges=ignored_adjacent_edges,
    )
    violating_pairs = separation["edges"] or []
    violating_distances = separation["distances"] or []

    masks: PolylineViolationMasks | None = None
    if include_violation_masks:
        separation_edges = np.zeros(len(polyline) - 1, dtype=np.bool_)
        for first_edge, second_edge in violating_pairs:
            separation_edges[first_edge] = True
            separation_edges[second_edge] = True
        curvature_vertices = np.zeros(len(polyline), dtype=np.bool_)
        curvature_vertices[1:-1] = curvature_inner_mask
        masks = PolylineViolationMasks(
            separation_edges=separation_edges,
            curvature_vertices=curvature_vertices,
        )

    return PolylineValidationResult(
        maximum_angle=float(np.max(angles, initial=np.float32(0.0))),
        curvature_violation_count=curvature_violation_count,
        separation_violation_count=len(violating_pairs),
        minimum_violating_distance=(
            min(violating_distances) if violating_distances else None
        ),
        ignored_adjacent_edges=ignored_adjacent_edges,
        violation_masks=masks,
    )
