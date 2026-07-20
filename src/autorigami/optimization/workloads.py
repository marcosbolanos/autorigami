from __future__ import annotations

from typing import Literal, TypeAlias, cast
import warnings

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from autorigami._native import apply_separation_correction
from autorigami.geometry.curvature import (
    get_polyline_angles,
)
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.geometry.separation import (
    DEFAULT_MIN_DISTANCE,
    check_self_intersections,
    get_candidate_intersecting_edges,
)
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
    target_angle: float,
    edge_length: float,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    local_exclusion_length: float | None = None,
    compression_fraction: float = 1e-2,
    minimum_compression_fraction: float = 1e-4,
    steps: int = 100,
    separation_steps: int = 80,
    separation_buffer: float = 8e-3,
    length_tolerance: float = 1e-2,
) -> Polyline:
    """Compress a polyline along one axis while preserving hard constraints.

    Each step contracts the selected coordinate, restores the original total
    arc length by scaling the perpendicular plane, and reparametrizes the
    curve. Existing separation correction supplies clearance before a final
    uniform length correction. A candidate is accepted only when total length,
    curvature, and nonlocal separation all validate. Rejected compression
    fractions are repeatedly halved; no invalid candidate is returned.

    Args:
        polyline: Float32 vertex coordinates with shape ``(n, 3)``. The input
            must already satisfy the curvature and separation constraints.
        axis: Coordinate whose total extent should be reduced.
        target_angle: Maximum turning angle in radians.
        edge_length: Arc-length sampling interval for the returned polyline.
        min_distance: Minimum distance between arc-nonlocal edges.
        local_exclusion_length: Centerline distance handled by the curvature
            constraint rather than separation. Defaults to ``min_distance``.
        compression_fraction: Fractional axis contraction proposed per step.
        minimum_compression_fraction: Smallest fraction tried after halving.
        steps: Maximum number of accepted compression steps.
        separation_steps: Correction passes allowed for each candidate.
        separation_buffer: Clearance reserved before final length restoration.
        length_tolerance: Absolute tolerance for total arc-length restoration.

    Returns:
        The last valid compressed float32 polyline.
    """
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    assert len(polyline) >= 3, "polyline must contain at least three points"
    assert axis in ("x", "y", "z"), "axis must be x, y, or z"
    assert 0.0 < target_angle < np.pi, "target_angle must be between zero and pi"
    assert edge_length > 0.0, "edge_length must be positive"
    assert min_distance > 0.0, "min_distance must be positive"
    assert 0.0 < compression_fraction < 1.0, (
        "compression_fraction must be between zero and one"
    )
    assert 0.0 < minimum_compression_fraction <= compression_fraction, (
        "minimum_compression_fraction must be positive and no larger than "
        "compression_fraction"
    )
    assert steps >= 0, "steps must be non-negative"
    assert separation_steps > 0, "separation_steps must be positive"
    assert separation_buffer > 0.0, "separation_buffer must be positive"
    assert length_tolerance > 0.0, "length_tolerance must be positive"

    exclusion_length = (
        min_distance if local_exclusion_length is None else local_exclusion_length
    )
    assert exclusion_length > 0.0, "local_exclusion_length must be positive"
    ignored_adjacent_edges = int(np.ceil(exclusion_length / edge_length))
    axis_index = ("x", "y", "z").index(axis)
    perpendicular_indices_by_axis: tuple[tuple[int, int], ...] = (
        (1, 2),
        (0, 2),
        (0, 1),
    )
    perpendicular_indices = perpendicular_indices_by_axis[axis_index]
    target_length = _polyline_length(polyline)
    compressed = np.array(polyline, dtype=np.float32, copy=True)
    _validate_axis_compression_constraints(
        compressed,
        target_length=target_length,
        target_angle=target_angle,
        min_distance=min_distance,
        ignored_adjacent_edges=ignored_adjacent_edges,
        length_tolerance=length_tolerance,
    )

    for _ in range(steps):
        current_fraction = compression_fraction
        accepted = False
        while current_fraction >= minimum_compression_fraction:
            candidate = _axis_compression_candidate(
                compressed,
                axis_index=axis_index,
                perpendicular_indices=perpendicular_indices,
                compression_fraction=current_fraction,
                target_length=target_length,
                edge_length=edge_length,
                min_distance=min_distance,
                ignored_adjacent_edges=ignored_adjacent_edges,
                separation_steps=separation_steps,
                separation_buffer=separation_buffer,
            )
            if _axis_compression_constraints_hold(
                candidate,
                target_length=target_length,
                target_angle=target_angle,
                min_distance=min_distance,
                ignored_adjacent_edges=ignored_adjacent_edges,
                length_tolerance=length_tolerance,
            ):
                compressed = candidate
                accepted = True
                break
            current_fraction *= 0.5
        if not accepted:
            break
    return compressed


def _axis_compression_candidate(
    polyline: Polyline,
    axis_index: int,
    perpendicular_indices: tuple[int, int],
    compression_fraction: float,
    target_length: float,
    edge_length: float,
    min_distance: float,
    ignored_adjacent_edges: int,
    separation_steps: int,
    separation_buffer: float,
) -> Polyline:
    candidate = np.array(polyline, dtype=np.float32, copy=True)
    axis_center = np.float32(
        0.5
        * (
            float(np.min(candidate[:, axis_index]))
            + float(np.max(candidate[:, axis_index]))
        )
    )
    candidate[:, axis_index] = axis_center + np.float32(1.0 - compression_fraction) * (
        candidate[:, axis_index] - axis_center
    )
    candidate = _restore_length_in_plane(
        candidate,
        coordinate_indices=perpendicular_indices,
        target_length=target_length,
    )
    candidate = reparametrize_arc_length(candidate, edge_length)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        candidate = fix_separation_violations(
            candidate,
            min_distance=min_distance,
            edge_length=edge_length,
            buffer=separation_buffer,
            steps=separation_steps,
            candidate_refresh_interval=10,
            n_ignored_adjacent_edges=ignored_adjacent_edges,
        )
    return _restore_total_length_uniformly(candidate, target_length)


def _restore_length_in_plane(
    polyline: Polyline,
    coordinate_indices: tuple[int, int],
    target_length: float,
) -> Polyline:
    center = np.mean(polyline[:, coordinate_indices], axis=0)
    centered = polyline[:, coordinate_indices] - center

    def length_error(scale: float) -> float:
        scaled = np.array(polyline, dtype=np.float32, copy=True)
        scaled[:, coordinate_indices] = center + np.float32(scale) * centered
        return _polyline_length(scaled) - target_length

    lower = 0.0
    upper = 1.0
    assert length_error(lower) <= 0.0, (
        "perpendicular coordinates cannot restore the target arc length"
    )
    while length_error(upper) < 0.0:
        upper *= 2.0
    scale = cast(float, brentq(length_error, lower, upper, xtol=1e-10))
    restored = np.array(polyline, dtype=np.float32, copy=True)
    restored[:, coordinate_indices] = center + np.float32(scale) * centered
    return restored


def _restore_total_length_uniformly(
    polyline: Polyline,
    target_length: float,
) -> Polyline:
    center = np.mean(polyline, axis=0)
    scale = np.float32(target_length / _polyline_length(polyline))
    return np.asarray(center + scale * (polyline - center), dtype=np.float32)


def _polyline_length(polyline: Polyline) -> float:
    return float(np.sum(np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)))


def _axis_compression_constraints_hold(
    polyline: Polyline,
    target_length: float,
    target_angle: float,
    min_distance: float,
    ignored_adjacent_edges: int,
    length_tolerance: float,
) -> bool:
    if abs(_polyline_length(polyline) - target_length) > length_tolerance:
        return False
    maximum_angle = float(np.max(get_polyline_angles(polyline)))
    if maximum_angle > target_angle + 1e-6:
        return False
    separation = check_self_intersections(
        polyline,
        min_euclid_distance=min_distance,
        n_ignored_adjacent_edges=ignored_adjacent_edges,
    )
    return separation["edges"] is None


def _validate_axis_compression_constraints(
    polyline: Polyline,
    target_length: float,
    target_angle: float,
    min_distance: float,
    ignored_adjacent_edges: int,
    length_tolerance: float,
) -> None:
    if not _axis_compression_constraints_hold(
        polyline,
        target_length=target_length,
        target_angle=target_angle,
        min_distance=min_distance,
        ignored_adjacent_edges=ignored_adjacent_edges,
        length_tolerance=length_tolerance,
    ):
        raise ValueError(
            "compress_along_axis requires an input polyline satisfying its "
            "length, curvature, and separation constraints"
        )


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
