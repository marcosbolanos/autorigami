from __future__ import annotations

# pyright: reportOptionalSubscript=false

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsmr

from autorigami.geometry.curvature import DEFAULT_MAX_ANGLE_RADIANS
from autorigami.geometry.separation import DEFAULT_MIN_DISTANCE
from autorigami.geometry.validation import validate_polyline
from autorigami.optimization.constraints import (
    ActiveConstraintJacobian,
    EqualityConstraintSystem,
    active_constraint_jacobian,
)
from autorigami.optimization.fractional_sobolev import FractionalSobolevPreconditioner
from autorigami.optimization.geometric_hessian import active_geometric_hessian_product
from autorigami.optimization.inextensibility import (
    edge_constraint_jacobian,
    edge_constraint_residuals,
    maximum_edge_length_error,
    reference_edge_lengths,
)
from autorigami.optimization.nonlinear_projection import project_nonlinear_constraints
from autorigami.optimization.projected_flow import solve_fractional_mixed_step
from autorigami.optimization.second_order import (
    axis_variance,
    axis_variance_gradient,
    lagrangian_hessian_product,
    lowest_tangent_mode,
)
from autorigami.types import Polyline

FloatArray = npt.NDArray[np.float64]
AxisName = Literal["x", "y", "z"]


@dataclass(frozen=True)
class AxisPackingIteration:
    iteration: int
    objective: float
    maximum_vertex_step: float
    edge_error: float
    barycenter_error: float
    minimum_constraint_slack: float
    maximum_angle: float
    separation_violations: int
    used_negative_curvature_mode: bool


@dataclass(frozen=True)
class AxisPackingResult:
    points: Polyline
    converged: bool
    message: str
    initial_objective: float
    final_objective: float
    iterations: tuple[AxisPackingIteration, ...]


def optimize_axis_packing(
    polyline: Polyline,
    *,
    axis: AxisName = "z",
    maximum_angle: float = DEFAULT_MAX_ANGLE_RADIANS,
    minimum_distance: float = DEFAULT_MIN_DISTANCE,
    ignored_adjacent_edges: int = 100,
    maximum_iterations: int = 10,
    maximum_vertex_step: float = 0.05,
    sobolev_sigma: float = 0.25,
) -> AxisPackingResult:
    """Minimize axis variance with exact edge and hard geometric constraints.

    The input sampling is the material discretization: each original edge
    length becomes a separate equality. The loop never rescales or
    reparametrizes the curve.
    """
    assert polyline.dtype == np.float32
    assert polyline.ndim == 2 and polyline.shape[1] == 3 and len(polyline) >= 3
    assert axis in ("x", "y", "z")
    assert 0.0 < maximum_angle < np.pi and minimum_distance > 0.0
    assert ignored_adjacent_edges >= 1 and maximum_iterations >= 0
    assert maximum_vertex_step > 0.0 and 0.0 < sobolev_sigma < 1.0
    initial_validation = validate_polyline(
        polyline,
        maximum_angle=maximum_angle,
        minimum_distance=minimum_distance,
        valid_curvature_ignored_adjacent_edges=ignored_adjacent_edges,
    )
    if not initial_validation.valid:
        raise ValueError(
            "axis packing requires a green input; got "
            f"{initial_validation.curvature_violation_count} curvature and "
            f"{initial_validation.separation_violation_count} separation violations"
        )
    axis_index = ("x", "y", "z").index(axis)
    points = np.asarray(polyline, dtype=np.float64)
    orientation_reference = np.array(points, copy=True)
    lengths = reference_edge_lengths(points)
    center = np.mean(points, axis=0)
    initial_objective = axis_variance(points, axis_index)
    history: list[AxisPackingIteration] = []
    trust_step = maximum_vertex_step
    message = "maximum iterations reached"
    converged = False

    def inequalities(candidate: FloatArray):
        return active_constraint_jacobian(
            candidate,
            target_angle=maximum_angle - 1e-6,
            min_distance=minimum_distance + 2e-4,
            ignored_adjacent_edges=ignored_adjacent_edges,
            contact_activation_distance=0.05,
            curvature_activation_angle=0.02,
        )

    for iteration in range(maximum_iterations):
        gradient = axis_variance_gradient(points, axis_index)
        equalities = _edge_and_rigid_equalities(points, lengths)
        geometric = inequalities(points)
        metric = FractionalSobolevPreconditioner(points, sigma=sobolev_sigma)
        try:
            direction_result = solve_fractional_mixed_step(
                gradient,
                metric=metric,
                equalities=equalities,
                inequalities=geometric,
                maximum_vertex_step=trust_step,
                safety_margin=1e-5,
                regularization=1e-8,
                relative_tolerance=1e-7,
                maximum_iterations=300,
            )
        except RuntimeError as error:
            message = str(error)
            break
        direction = direction_result.displacement
        used_negative_mode = False
        if float(np.max(np.linalg.norm(direction, axis=1), initial=0.0)) < 1e-5:
            active_geometric = geometric.select(direction_result.active_constraint_indices)
            edge_multipliers, geometric_multipliers = _stationary_multipliers(
                points,
                lengths,
                gradient,
                active_geometric,
            )

            def hessian(displacement: FloatArray) -> FloatArray:
                def geometric_product(local_direction: FloatArray) -> FloatArray:
                    return active_geometric_hessian_product(
                        points,
                        active_geometric,
                        geometric_multipliers,
                        local_direction,
                    )

                return lagrangian_hessian_product(
                    displacement,
                    axis=axis_index,
                    edge_multipliers=edge_multipliers,
                    geometric_hessian_product=(
                        geometric_product
                        if active_geometric.constraint_count
                        else None
                    ),
                )

            mode = lowest_tangent_mode(points, lengths, hessian)
            if not mode.is_negative:
                converged = True
                message = "first- and second-order constrained stationarity reached"
                break
            direction = trust_step * mode.displacement
            used_negative_mode = True
        projection = project_nonlinear_constraints(
            points + direction,
            lengths,
            inequalities,
            edge_tolerance=1e-8,
            inequality_tolerance=1e-6,
        )
        if not projection.converged:
            trust_step *= 0.5
            if trust_step < 1e-5:
                message = projection.message
                break
            continue
        candidate = _align_global_orientation(
            projection.points,
            orientation_reference,
            center,
        )
        candidate32 = np.asarray(candidate, dtype=np.float32)
        validation = validate_polyline(
            candidate32,
            maximum_angle=maximum_angle,
            minimum_distance=minimum_distance,
            valid_curvature_ignored_adjacent_edges=ignored_adjacent_edges,
        )
        candidate_objective = axis_variance(candidate, axis_index)
        if not validation.valid or candidate_objective >= axis_variance(points, axis_index):
            trust_step *= 0.5
            if trust_step < 1e-5:
                message = "trust region exhausted without a valid improving projection"
                break
            continue
        points = candidate
        history.append(
            AxisPackingIteration(
                iteration=iteration,
                objective=candidate_objective,
                maximum_vertex_step=trust_step,
                edge_error=maximum_edge_length_error(points, lengths),
                barycenter_error=float(np.linalg.norm(np.mean(points, axis=0) - center)),
                minimum_constraint_slack=projection.minimum_slack,
                maximum_angle=validation.maximum_angle,
                separation_violations=validation.separation_violation_count,
                used_negative_curvature_mode=used_negative_mode,
            )
        )
        trust_step = min(maximum_vertex_step, 1.2 * trust_step)

    result = np.asarray(points, dtype=np.float32)
    final_validation = validate_polyline(
        result,
        maximum_angle=maximum_angle,
        minimum_distance=minimum_distance,
        valid_curvature_ignored_adjacent_edges=ignored_adjacent_edges,
    )
    if not final_validation.valid:
        raise RuntimeError("axis packing produced a result rejected by standard validation")
    return AxisPackingResult(
        points=result,
        converged=converged,
        message=message,
        initial_objective=initial_objective,
        final_objective=axis_variance(points, axis_index),
        iterations=tuple(history),
    )


def _edge_and_rigid_equalities(points: FloatArray, lengths: FloatArray) -> EqualityConstraintSystem:
    edge_jacobian = edge_constraint_jacobian(points, lengths)
    centered = points - np.mean(points, axis=0)
    rows: list[FloatArray] = []
    for coordinate in range(3):
        translation = np.zeros_like(points)
        translation[:, coordinate] = 1.0 / np.sqrt(len(points))
        rows.append(translation.reshape(-1))
    for axis in np.eye(3):
        rotation = np.asarray(
            np.cross(np.broadcast_to(axis, centered.shape), centered),
            dtype=np.float64,
        ).reshape(-1)
        norm = np.linalg.norm(rotation)
        if norm > 1e-12:
            rows.append(rotation / norm)
    rigid = csr_matrix(np.vstack(rows))
    jacobian = csr_matrix(vstack((edge_jacobian, rigid), format="csr"))
    values = np.concatenate((edge_constraint_residuals(points, lengths), np.zeros(len(rows))))
    return EqualityConstraintSystem(values=values, jacobian=jacobian, vertex_count=len(points))


def _stationary_multipliers(
    points: FloatArray,
    lengths: FloatArray,
    gradient: FloatArray,
    geometric: ActiveConstraintJacobian,
) -> tuple[FloatArray, FloatArray]:
    edge = edge_constraint_jacobian(points, lengths)
    matrix = csr_matrix(vstack((edge, geometric.matrix), format="csr"))
    solution = lsmr(
        matrix.T,
        -gradient.reshape(-1),
        atol=1e-11,
        btol=1e-11,
        maxiter=max(100, 4 * matrix.shape[0]),
    )
    if solution[1] not in (1, 2):  # pyright: ignore[reportOptionalSubscript]
        raise RuntimeError("stationary multiplier solve did not converge")
    multipliers = np.asarray(  # pyright: ignore[reportOptionalSubscript]
        solution[0], dtype=np.float64
    )
    return multipliers[: len(lengths)], multipliers[len(lengths) :]


def _align_global_orientation(
    points: FloatArray,
    reference: FloatArray,
    center: FloatArray,
) -> FloatArray:
    """Remove the best-fit global rotation and restore the fixed barycenter."""
    centered = points - np.mean(points, axis=0)
    reference_centered = reference - np.mean(reference, axis=0)
    left, _, right_transpose = np.linalg.svd(centered.T @ reference_centered)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return centered @ rotation + center
