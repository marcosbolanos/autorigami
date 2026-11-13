"""Gradient repair of turning-angle violations."""

from __future__ import annotations

from typing import TypeAlias
import warnings

import numpy as np
import numpy.typing as npt

from autorigami.geometry.curvature import get_polyline_angles
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.types import Polyline

CoordinateMask: TypeAlias = tuple[bool, bool, bool]

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
