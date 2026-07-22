from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, vstack

from autorigami.gravity.constraints import (
    GravityConstraintConfiguration,
    GravityConstraintResiduals,
    evaluate_gravity_constraint_residuals,
    linearize_geometric_constraints,
)
from autorigami.gravity.quadratic_step import solve_minimum_norm_correction
from autorigami.optimization.inextensibility import (
    edge_constraint_jacobian,
    edge_constraint_residuals,
    maximum_edge_length_error,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GravityProjectionConfiguration:
    edge_tolerance: float = 1e-7
    inequality_tolerance: float = 1e-7
    validation_safety_margin: float = 2e-5
    maximum_iterations: int = 20
    maximum_coordinate_correction: float = 0.2

    def __post_init__(self) -> None:
        assert self.edge_tolerance > 0.0
        assert self.inequality_tolerance >= 0.0
        assert self.validation_safety_margin > 0.0
        assert self.maximum_iterations >= 1
        assert self.maximum_coordinate_correction > 0.0


@dataclass(frozen=True)
class GravityProjectionIteration:
    iteration: int
    edge_error: float
    minimum_curvature_slack: float
    minimum_contact_slack: float
    minimum_floor_slack: float
    curvature_count: int
    contact_count: int
    active_inequality_count: int


@dataclass(frozen=True)
class GravityProjectionResult:
    points: FloatArray
    converged: bool
    iterations: int
    residuals: GravityConstraintResiduals
    minimum_margin_slack: float
    message: str
    history: tuple[GravityProjectionIteration, ...]


def project_gravity_constraints(
    candidate: FloatArray,
    reference_lengths: FloatArray,
    *,
    floor_height: float,
    tube_radius: float,
    constraint_configuration: GravityConstraintConfiguration,
    projection_configuration: GravityProjectionConfiguration,
) -> GravityProjectionResult:
    """Project onto coupled nonlinear chain and geometric constraints."""
    assert candidate.dtype == np.float64
    assert candidate.ndim == 2 and candidate.shape[1] == 3
    assert reference_lengths.dtype == np.float64
    assert reference_lengths.shape == (len(candidate) - 1,)
    points = np.array(candidate, copy=True)
    minimum_margin_slack = np.inf
    history: list[GravityProjectionIteration] = []

    for iteration in range(projection_configuration.maximum_iterations + 1):
        edge_matrix = edge_constraint_jacobian(points, reference_lengths)
        edge_rhs = -edge_constraint_residuals(points, reference_lengths)
        geometric = linearize_geometric_constraints(
            points,
            floor_height=floor_height,
            tube_radius=tube_radius,
            contact_search_radius=(
                constraint_configuration.minimum_distance
                + constraint_configuration.contact_safety_margin
            ),
            configuration=constraint_configuration,
            include_all_curvatures=False,
        )
        physical_slacks = _physical_geometric_slacks(
            geometric.slacks,
            curvature_count=geometric.curvature_count,
            contact_count=geometric.contact_count,
            configuration=constraint_configuration,
        )
        required_slacks = (
            physical_slacks - projection_configuration.validation_safety_margin
        )
        active_rows = np.flatnonzero(
            required_slacks < -projection_configuration.inequality_tolerance
        ).astype(np.int64)
        edge_error = maximum_edge_length_error(points, reference_lengths)
        minimum_margin_slack = float(
            np.min(geometric.slacks, initial=np.inf)
        )
        curvature_end = geometric.curvature_count
        contact_end = curvature_end + geometric.contact_count
        history.append(
            GravityProjectionIteration(
                iteration=iteration,
                edge_error=edge_error,
                minimum_curvature_slack=float(
                    np.min(geometric.slacks[:curvature_end], initial=np.inf)
                ),
                minimum_contact_slack=float(
                    np.min(
                        geometric.slacks[curvature_end:contact_end],
                        initial=np.inf,
                    )
                ),
                minimum_floor_slack=float(
                    np.min(geometric.slacks[contact_end:], initial=np.inf)
                ),
                curvature_count=geometric.curvature_count,
                contact_count=geometric.contact_count,
                active_inequality_count=len(active_rows),
            )
        )
        if (
            edge_error <= projection_configuration.edge_tolerance
            and len(active_rows) == 0
        ):
            return _projection_result(
                points,
                reference_lengths,
                floor_height=floor_height,
                tube_radius=tube_radius,
                constraint_configuration=constraint_configuration,
                converged=True,
                iterations=iteration,
                minimum_margin_slack=minimum_margin_slack,
                message="nonlinear gravity constraints converged",
                history=history,
            )
        if iteration == projection_configuration.maximum_iterations:
            break

        active_matrix = geometric.matrix[active_rows]
        correction_matrix = csc_matrix(
            vstack((edge_matrix, active_matrix), format="csc")
        )
        correction_lower = np.concatenate(
            (edge_rhs, -required_slacks[active_rows])
        )
        correction_upper = np.concatenate(
            (edge_rhs, np.full(len(active_rows), np.inf))
        )
        correction = solve_minimum_norm_correction(
            correction_matrix,
            np.asarray(correction_lower, dtype=np.float64),
            np.asarray(correction_upper, dtype=np.float64),
            vertex_count=len(points),
            maximum_coordinate_correction=(
                projection_configuration.maximum_coordinate_correction
            ),
        )
        if not correction.succeeded:
            return _projection_result(
                points,
                reference_lengths,
                floor_height=floor_height,
                tube_radius=tube_radius,
                constraint_configuration=constraint_configuration,
                converged=False,
                iterations=iteration,
                minimum_margin_slack=minimum_margin_slack,
                message=(
                    "active-set projection failed with status "
                    f"{correction.status}"
                ),
                history=history,
            )
        assert correction.displacement is not None
        points += correction.displacement

    return _projection_result(
        points,
        reference_lengths,
        floor_height=floor_height,
        tube_radius=tube_radius,
        constraint_configuration=constraint_configuration,
        converged=False,
        iterations=projection_configuration.maximum_iterations,
        minimum_margin_slack=minimum_margin_slack,
        message="nonlinear gravity projection did not converge",
        history=history,
    )


def _physical_geometric_slacks(
    margin_slacks: FloatArray,
    *,
    curvature_count: int,
    contact_count: int,
    configuration: GravityConstraintConfiguration,
) -> FloatArray:
    """Remove QP activation margins while retaining physical slacks."""
    physical = np.array(margin_slacks, copy=True)
    curvature_end = curvature_count
    contact_end = curvature_end + contact_count
    physical[:curvature_end] += configuration.curvature_safety_margin
    physical[curvature_end:contact_end] += configuration.contact_safety_margin
    physical[contact_end:] += configuration.floor_safety_margin
    return physical


def _projection_result(
    points: FloatArray,
    reference_lengths: FloatArray,
    *,
    floor_height: float,
    tube_radius: float,
    constraint_configuration: GravityConstraintConfiguration,
    converged: bool,
    iterations: int,
    minimum_margin_slack: float,
    message: str,
    history: list[GravityProjectionIteration],
) -> GravityProjectionResult:
    residuals = evaluate_gravity_constraint_residuals(
        points,
        reference_lengths,
        floor_height=floor_height,
        tube_radius=tube_radius,
        configuration=constraint_configuration,
    )
    return GravityProjectionResult(
        points=points,
        converged=converged,
        iterations=iterations,
        residuals=residuals,
        minimum_margin_slack=minimum_margin_slack,
        message=message,
        history=tuple(history),
    )
