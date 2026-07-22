from __future__ import annotations

from typing import Literal, TypeAlias
import warnings

import numpy as np
import numpy.typing as npt

from autorigami._native import apply_separation_correction
from autorigami.geometry.curvature import (
    DEFAULT_EDGE_LENGTH_NM,
    DEFAULT_MAX_ANGLE_RADIANS,
    get_polyline_angles,
)
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.geometry.separation import (
    DEFAULT_MIN_DISTANCE,
    check_self_intersections,
    get_candidate_intersecting_edges,
)
from autorigami.optimization.axis_packing import optimize_axis_packing
from autorigami.optimization.energies import curvature_violation_energy_gradient
from autorigami.optimization.separation import (
    optimize_separation as optimize_separation,
)
from autorigami.types import Polyline

CoordinateMask: TypeAlias = tuple[bool, bool, bool]
CoordinateName: TypeAlias = Literal["x", "y", "z"]

def compress_along_axis(
    polyline: Polyline,
    *,
    axis: CoordinateName,
    target_angle: float = DEFAULT_MAX_ANGLE_RADIANS,
    edge_length: float = DEFAULT_EDGE_LENGTH_NM,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    local_exclusion_length: float = 34.0,
    steps: int = 10,
    maximum_vertex_step: float = 5e-2,
) -> Polyline:
    """Pack along an axis while preserving each material edge exactly.

    This compatibility workload delegates to the constrained implementation;
    ``edge_length`` is used only to translate the physical local-exclusion
    length into an edge-index gap. No resampling occurs in the optimizer.
    """
    assert edge_length > 0.0 and local_exclusion_length > 0.0
    ignored_adjacent_edges = int(np.ceil(local_exclusion_length / edge_length))
    result = optimize_axis_packing(
        polyline,
        axis=axis,
        maximum_angle=target_angle,
        minimum_distance=min_distance,
        ignored_adjacent_edges=ignored_adjacent_edges,
        maximum_iterations=steps,
        maximum_vertex_step=maximum_vertex_step,
    )
    return result.points


def fix_separation_violations(
    polyline: Polyline,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    *,
    edge_length: float,
    coordinates: list[CoordinateName] = ["x", "y", "z"],
    fixed_step: float = 1e-3,
    buffer: float = 5e-3,
    steps: int = 100,
    candidate_refresh_interval: int = 5,
    n_ignored_adjacent_edges: int = 1,
    validation_attempts: int = 3,
) -> Polyline:
    """Correct separation violations and restore arc-length parametrization.

    Candidate pairs are corrected before the curve is reparametrized and
    validated with ``check_self_intersections``. A failed validation triggers
    another attempt with one additional ``buffer`` of clearance. The function
    warns before returning a curve that remains invalid.

    Args:
        polyline: Float32 vertex coordinates with shape ``(n, 3)``.
        min_distance: Required distance between non-ignored edge pairs, in the
            same units as ``polyline``.
        edge_length: Arc-length sampling interval for the returned polyline.
        coordinates: Coordinate names that separation corrections may change.
        fixed_step: Constant distance added to each active pair correction.
        buffer: Extra separation reserved for each validation attempt.
        steps: Maximum number of native correction passes.
        candidate_refresh_interval: Passes performed before rebuilding KD-tree
            candidates.
        n_ignored_adjacent_edges: Maximum edge-index difference excluded from
            separation checks.
        validation_attempts: Correction and validation attempts before warning
            and returning a remaining invalid result.

    Returns:
        A new float32 polyline sampled at ``edge_length`` arc-length intervals;
        its final edge may be shorter.
    """
    optimized_polyline = np.array(polyline, dtype=np.float32, copy=True)
    coordinate_mask: CoordinateMask = (
        "x" in coordinates,
        "y" in coordinates,
        "z" in coordinates,
    )
    remaining_edges: list[tuple[int, int]] = []
    remaining_distances: list[float] = []

    for attempt in range(1, validation_attempts + 1):
        correction_distance = min_distance + attempt * buffer
        native_correction_distance = float(
            np.nextafter(np.float32(correction_distance), np.float32(np.inf))
        )
        completed_steps = 0
        while completed_steps < steps:
            pass_count = min(candidate_refresh_interval, steps - completed_steps)
            candidate_pairs = get_candidate_intersecting_edges(
                optimized_polyline,
                min_distance=correction_distance,
                n_ignored_adjacent_edges=n_ignored_adjacent_edges,
            )
            corrected_polyline, correction_count = apply_separation_correction(
                optimized_polyline,
                pass_count,
                candidate_pairs,
                native_correction_distance,
                fixed_step,
                coordinate_mask,
                reverse_order=bool(completed_steps % 2),
            )
            optimized_polyline = np.asarray(corrected_polyline, dtype=np.float32)
            if correction_count == 0:
                break
            completed_steps += pass_count

        optimized_polyline = reparametrize_arc_length(
            optimized_polyline,
            edge_length,
        )
        validation = check_self_intersections(
            optimized_polyline,
            min_euclid_distance=min_distance,
            n_ignored_adjacent_edges=n_ignored_adjacent_edges,
        )
        if validation["edges"] is None:
            return optimized_polyline
        assert validation["distances"] is not None
        remaining_edges = validation["edges"]
        remaining_distances = validation["distances"]

    warnings.warn(
        "fix_separation_violations is returning an invalid polyline after "
        f"{validation_attempts} attempts: {len(remaining_edges)} separation "
        f"violations remain; minimum distance is {min(remaining_distances)}",
        RuntimeWarning,
        stacklevel=2,
    )
    return optimized_polyline


def fix_curvature_violations(
    polyline: Polyline,
    target_angle: float,
    edge_length: float,
    coordinate_mask: CoordinateMask = (True, True, True),
    learning_rate: float = 5e-3,
    steps: int = 500,
    max_vertex_step: float = 0.05,
    reparametrize_every: int = 1,
    validation_attempts: int = 3,
) -> Polyline:
    """Reduce turning-angle violations with gradient descent.

    Every inner vertex whose turning angle exceeds ``target_angle`` contributes
    to a quadratic violation energy. Each attempt takes clipped negative
    gradient steps and then validates every turning angle. Invalid results are
    optimized again. The function warns before returning a curve that remains
    invalid after ``validation_attempts``.

    Args:
        polyline: Float32 vertex coordinates with shape ``(n, 3)``.
        target_angle: Maximum desired turning angle in radians.
        coordinate_mask: Coordinates that gradient updates may change.
        learning_rate: Multiplier applied to each clipped gradient update.
        steps: Maximum number of gradient iterations.
        max_vertex_step: Maximum norm of an unscaled per-vertex update.
        edge_length: Optional arc-length sampling interval. Pass ``None`` to
            preserve the existing vertices without reparametrization.
        reparametrize_every: Iterations between reparametrizations when
            ``edge_length`` is provided.
        validation_attempts: Optimization and validation attempts before
            warning and returning a remaining invalid result.

    Returns:
        A new float32 polyline satisfying the target angle when validation
        succeeds, or the remaining result after a warning.
    """
    optimized_polyline = np.array(polyline, dtype=np.float32, copy=True)
    coordinate_mask_array = np.array(coordinate_mask, dtype=np.float32)

    violation_count = 0
    maximum_angle = 0.0
    for _ in range(validation_attempts):
        for step in range(steps):
            update = _curvature_update(
                polyline=optimized_polyline,
                target_angle=target_angle,
                max_vertex_step=max_vertex_step,
            )
            if not np.any(update):
                break

            optimized_polyline += (
                np.float32(learning_rate) * update * coordinate_mask_array
            )
            if edge_length is not None and (step + 1) % reparametrize_every == 0:
                optimized_polyline = reparametrize_arc_length(
                    optimized_polyline,
                    edge_length,
                )

        angles = get_polyline_angles(optimized_polyline)
        violations = angles > np.float32(target_angle)
        violation_count = int(np.count_nonzero(violations))
        maximum_angle = float(np.max(angles, initial=np.float32(0.0)))
        if violation_count == 0:
            return optimized_polyline

    warnings.warn(
        "fix_curvature_violations is returning an invalid polyline after "
        f"{validation_attempts} attempts: {violation_count} curvature "
        f"violations remain; maximum angle is {maximum_angle}",
        RuntimeWarning,
        stacklevel=2,
    )
    return optimized_polyline


def _curvature_update(
    polyline: Polyline,
    target_angle: float,
    max_vertex_step: float,
) -> Polyline:
    _EPSILON = np.float32(1e-12)
    energy_gradient = curvature_violation_energy_gradient(polyline, target_angle)
    return _clip_vertex_steps(-energy_gradient, max_vertex_step)


def _clip_vertex_steps(
    updates: npt.NDArray[np.float32],
    max_vertex_step: float,
) -> npt.NDArray[np.float32]:
    _EPSILON = np.float32(1e-12)
    step_norms = np.linalg.norm(updates, axis=1)
    scales = np.minimum(
        1.0,
        np.float32(max_vertex_step) / np.maximum(step_norms, _EPSILON),
    )
    return updates * scales[:, None]
