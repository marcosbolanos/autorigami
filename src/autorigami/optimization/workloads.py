from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from autorigami._native import segment_segment_distance
from autorigami.geometry.curvature import get_polyline_angles
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.geometry.separation import get_candidate_intersecting_edges
from autorigami.types import EdgeIndex, Polyline, Vector3

CoordinateMask: TypeAlias = tuple[bool, bool, bool]

_EPSILON = np.float32(1e-12)


def optimize_separation_violations(
    polyline: Polyline,
    min_distance: float,
    coordinate_mask: CoordinateMask = (True, True, True),
    learning_rate: float = 1e-2,
    steps: int = 100,
    n_ignored_adjacent_edges: int = 1,
) -> Polyline:
    optimized_polyline = np.array(polyline, dtype=np.float32, copy=True)
    coordinate_mask_array = np.array(coordinate_mask, dtype=np.float32)

    for _ in range(steps):
        candidate_pairs = get_candidate_intersecting_edges(
            optimized_polyline,
            min_distance=min_distance,
            n_ignored_adjacent_edges=n_ignored_adjacent_edges,
        )
        update = _separation_update(
            polyline=optimized_polyline,
            candidate_pairs=candidate_pairs,
            min_distance=min_distance,
        )
        if not np.any(update):
            break

        optimized_polyline += np.float32(learning_rate) * update * coordinate_mask_array

    return optimized_polyline


def fix_curvature_violations(
    polyline: Polyline,
    target_angle: float,
    coordinate_mask: CoordinateMask = (True, True, True),
    learning_rate: float = 0.2,
    steps: int = 100,
    max_vertex_step: float = 0.05,
    edge_length: float | None = None,
    reparametrize_every: int = 1,
) -> Polyline:
    optimized_polyline = np.array(polyline, dtype=np.float32, copy=True)
    coordinate_mask_array = np.array(coordinate_mask, dtype=np.float32)

    for step in range(steps):
        update = _curvature_update(
            polyline=optimized_polyline,
            target_angle=target_angle,
            max_vertex_step=max_vertex_step,
        )
        if not np.any(update):
            break

        optimized_polyline += np.float32(learning_rate) * update * coordinate_mask_array
        if (
            edge_length is not None
            and (step + 1) % reparametrize_every == 0
        ):
            optimized_polyline = reparametrize_arc_length(
                optimized_polyline,
                edge_length,
            )

    return optimized_polyline


def _curvature_update(
    polyline: Polyline,
    target_angle: float,
    max_vertex_step: float,
) -> Polyline:
    angles = get_polyline_angles(polyline)
    violations = np.maximum(np.float32(0.0), angles - np.float32(target_angle))
    violating_vertices = violations > 0.0

    update = np.zeros_like(polyline, dtype=np.float32)
    if not np.any(violating_vertices):
        return update

    neighbor_midpoints = np.float32(0.5) * (polyline[:-2] + polyline[2:])
    raw_updates = neighbor_midpoints - polyline[1:-1]
    weighted_updates = (
        raw_updates * (violations / np.float32(target_angle))[:, None]
    ).astype(np.float32)
    clipped_updates = _clip_vertex_steps(weighted_updates, max_vertex_step)
    inner_update = update[1:-1]
    inner_update[violating_vertices] = clipped_updates[violating_vertices]
    return update


def _clip_vertex_steps(
    updates: npt.NDArray[np.float32],
    max_vertex_step: float,
) -> npt.NDArray[np.float32]:
    step_norms = np.linalg.norm(updates, axis=1)
    scales = np.minimum(
        1.0,
        np.float32(max_vertex_step) / np.maximum(step_norms, _EPSILON),
    )
    return updates * scales[:, None]


def _separation_update(
    polyline: Polyline,
    candidate_pairs: list[tuple[EdgeIndex, EdgeIndex]],
    min_distance: float,
) -> Polyline:
    update = np.zeros_like(polyline, dtype=np.float32)
    distances = segment_segment_distance(
        polyline,
        candidate_pairs,
    )
    violating_candidate_pairs = [
        candidate_pair
        for candidate_pair, (distance, _, _) in zip(candidate_pairs, distances, strict=True)
        if distance < min_distance
    ]
    violation_distances = segment_segment_distance(
        polyline,
        violating_candidate_pairs,
        include_optimization_data=True,
    )

    for (first_index, second_index), distance_data in zip(
        violating_candidate_pairs,
        violation_distances,
        strict=True,
    ):
        distance, closest_first, closest_second, first_parameter, second_parameter = distance_data
        direction = _distance_increase_direction(
            polyline=polyline,
            first_index=first_index,
            second_index=second_index,
            closest_first=closest_first,
            closest_second=closest_second,
            distance=np.float32(distance),
        )
        violation = np.float32(min_distance - distance)
        _accumulate_segment_pair_update(
            update=update,
            first_index=first_index,
            second_index=second_index,
            first_parameter=np.float32(first_parameter),
            second_parameter=np.float32(second_parameter),
            direction=violation * direction,
        )

    return update


def _distance_increase_direction(
    polyline: Polyline,
    first_index: EdgeIndex,
    second_index: EdgeIndex,
    closest_first: Vector3,
    closest_second: Vector3,
    distance: np.float32,
) -> Vector3:
    closest_delta = closest_first - closest_second
    if distance > _EPSILON:
        return closest_delta / distance

    first_direction = polyline[first_index + 1] - polyline[first_index]
    second_direction = polyline[second_index + 1] - polyline[second_index]
    normal = np.cross(first_direction, second_direction)
    normal_norm = np.linalg.norm(normal)
    if normal_norm > _EPSILON:
        return normal / normal_norm

    first_midpoint = 0.5 * (polyline[first_index] + polyline[first_index + 1])
    second_midpoint = 0.5 * (polyline[second_index] + polyline[second_index + 1])
    midpoint_delta = first_midpoint - second_midpoint
    midpoint_delta_norm = np.linalg.norm(midpoint_delta)
    if midpoint_delta_norm > _EPSILON:
        return midpoint_delta / midpoint_delta_norm

    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _accumulate_segment_pair_update(
    update: npt.NDArray[np.float32],
    first_index: EdgeIndex,
    second_index: EdgeIndex,
    first_parameter: np.float32,
    second_parameter: np.float32,
    direction: Vector3,
) -> None:
    update[first_index] += (np.float32(1.0) - first_parameter) * direction
    update[first_index + 1] += first_parameter * direction
    update[second_index] -= (np.float32(1.0) - second_parameter) * direction
    update[second_index + 1] -= second_parameter * direction
