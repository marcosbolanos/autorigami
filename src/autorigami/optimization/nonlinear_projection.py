from __future__ import annotations

# pyright: reportOptionalSubscript=false

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsmr, spsolve

from autorigami.optimization.constraints import ActiveConstraintJacobian
from autorigami.optimization.inextensibility import (
    edge_constraint_jacobian,
    edge_constraint_residuals,
    maximum_edge_length_error,
)

FloatArray = npt.NDArray[np.float64]
InequalityFactory = Callable[[FloatArray], ActiveConstraintJacobian]


@dataclass(frozen=True)
class ProjectionResult:
    """Result and diagnostics of a nonlinear constraint projection."""

    points: FloatArray
    converged: bool
    iterations: int
    edge_error: float
    minimum_slack: float
    barycenter_error: float
    message: str


def project_chain_lengths(
    candidate: FloatArray,
    lengths: FloatArray,
    *,
    tolerance: float = 1e-10,
    maximum_iterations: int = 30,
) -> ProjectionResult:
    """Euclidean Newton projection onto every individual edge length.

    The sparse chain Jacobian is solved directly as a least-norm correction.
    This is an exact nonlinear projection: constraints are rebuilt after every
    correction, and no global scaling or resampling is performed.
    """
    assert candidate.dtype == np.float64
    assert lengths.dtype == np.float64
    assert tolerance > 0.0 and maximum_iterations > 0
    points = np.array(candidate, copy=True)
    target_center = np.mean(candidate, axis=0)
    for iteration in range(maximum_iterations + 1):
        error = maximum_edge_length_error(points, lengths)
        if error <= tolerance:
            points += target_center - np.mean(points, axis=0)
            return ProjectionResult(
                points=points,
                converged=True,
                iterations=iteration,
                edge_error=maximum_edge_length_error(points, lengths),
                minimum_slack=np.inf,
                barycenter_error=float(np.linalg.norm(np.mean(points, axis=0) - target_center)),
                message="all edge equalities converged",
            )
        if iteration == maximum_iterations:
            break
        jacobian = edge_constraint_jacobian(points, lengths)
        residuals = edge_constraint_residuals(points, lengths)
        normal = csr_matrix(jacobian @ jacobian.T)
        multipliers = np.asarray(spsolve(normal, -residuals), dtype=np.float64)
        correction = np.asarray(jacobian.T @ multipliers, dtype=np.float64)
        points += correction.reshape((-1, 3))
        points += target_center - np.mean(points, axis=0)
    return ProjectionResult(
        points=points,
        converged=False,
        iterations=maximum_iterations,
        edge_error=maximum_edge_length_error(points, lengths),
        minimum_slack=np.inf,
        barycenter_error=float(np.linalg.norm(np.mean(points, axis=0) - target_center)),
        message="edge equality projection did not converge",
    )


def project_nonlinear_constraints(
    candidate: FloatArray,
    lengths: FloatArray,
    inequality_factory: InequalityFactory,
    *,
    edge_tolerance: float = 1e-6,
    inequality_tolerance: float = 1e-6,
    maximum_iterations: int = 40,
) -> ProjectionResult:
    """SQP projection onto edge equalities and geometric lower bounds.

    Only violated inequalities are imposed in each sparse linearized solve.
    Every iteration rebuilds both constraint values and Jacobians, so the
    returned result is judged by nonlinear residuals rather than by a single
    tangent-plane approximation.
    """
    assert candidate.dtype == np.float64 and lengths.dtype == np.float64
    assert edge_tolerance > 0.0 and inequality_tolerance >= 0.0
    target_center = np.mean(candidate, axis=0)
    points = np.array(candidate, copy=True)
    minimum_slack = np.inf
    for iteration in range(maximum_iterations + 1):
        edge_error = maximum_edge_length_error(points, lengths)
        inequalities = inequality_factory(points)
        minimum_slack = float(np.min(inequalities.slacks, initial=np.inf))
        if edge_error <= edge_tolerance and minimum_slack >= -inequality_tolerance:
            return ProjectionResult(
                points=points,
                converged=True,
                iterations=iteration,
                edge_error=edge_error,
                minimum_slack=minimum_slack,
                barycenter_error=float(np.linalg.norm(np.mean(points, axis=0) - target_center)),
                message="edge equalities and inequalities converged",
            )
        if iteration == maximum_iterations:
            break
        edge_jacobian = edge_constraint_jacobian(points, lengths)
        edge_rhs = -edge_constraint_residuals(points, lengths)
        violated = np.flatnonzero(inequalities.slacks < 0.0).astype(np.int64)
        if len(violated):
            geometric = inequalities.select(violated)
            jacobian = vstack((edge_jacobian, geometric.matrix), format="csr")
            rhs = np.concatenate((edge_rhs, -geometric.slacks))
        else:
            jacobian = edge_jacobian
            rhs = edge_rhs
        correction = _least_norm_correction(jacobian, rhs).reshape((-1, 3))
        largest = float(np.max(np.linalg.norm(correction, axis=1), initial=0.0))
        if largest > 0.25:
            correction *= 0.25 / largest
        points += correction
        points += target_center - np.mean(points, axis=0)
    return ProjectionResult(
        points=points,
        converged=False,
        iterations=maximum_iterations,
        edge_error=maximum_edge_length_error(points, lengths),
        minimum_slack=minimum_slack,
        barycenter_error=float(np.linalg.norm(np.mean(points, axis=0) - target_center)),
        message="combined nonlinear projection did not converge",
    )


def _least_norm_correction(jacobian: csr_matrix, right_hand_side: FloatArray) -> FloatArray:
    assert jacobian.shape[0] == len(right_hand_side)
    if len(right_hand_side) == 0:
        return np.zeros(jacobian.shape[1], dtype=np.float64)
    solution = lsmr(
        jacobian,
        right_hand_side,
        atol=1e-12,
        btol=1e-12,
        maxiter=max(100, 4 * jacobian.shape[0]),
    )
    if solution[1] not in (1, 2):  # pyright: ignore[reportOptionalSubscript]
        raise RuntimeError(  # pyright: ignore[reportOptionalSubscript]
            f"sparse projection solve failed with LSMR status {solution[1]}"
        )
    return np.asarray(solution[0], dtype=np.float64)  # pyright: ignore[reportOptionalSubscript]
