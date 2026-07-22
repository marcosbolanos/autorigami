from __future__ import annotations

# pyright: reportOptionalSubscript=false

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import osqp
from scipy.sparse import csc_matrix, eye, vstack

from autorigami.gravity.constraints import GravityConstraintLinearization

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GravityQuadraticStepConfiguration:
    proximal_step: float = 0.05
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-6
    maximum_iterations: int = 10_000
    polishing: bool = True

    def __post_init__(self) -> None:
        assert self.proximal_step > 0.0
        assert self.absolute_tolerance > 0.0
        assert self.relative_tolerance > 0.0
        assert self.maximum_iterations >= 1


@dataclass(frozen=True)
class GravityQuadraticStepResult:
    displacement: FloatArray | None
    status: str
    succeeded: bool
    objective: float
    primal_residual: float
    dual_residual: float
    setup_time: float
    solve_time: float
    iterations: int
    warm_started: bool


def gravity_linear_cost(vertex_count: int) -> FloatArray:
    """Return the differential of total gravitational potential."""
    assert vertex_count >= 1
    cost = np.zeros((vertex_count, 3), dtype=np.float64)
    cost[:, 2] = 1.0
    return cost.reshape(-1)


def proximal_hessian(vertex_count: int, proximal_step: float) -> csc_matrix:
    """Return the strictly positive diagonal proximal QP Hessian."""
    assert vertex_count >= 1
    assert proximal_step > 0.0
    return csc_matrix(
        eye(3 * vertex_count, format="csc", dtype=np.float64) / proximal_step
    )


def solve_gravity_quadratic_step(
    linearization: GravityConstraintLinearization,
    *,
    configuration: GravityQuadraticStepConfiguration,
    warm_start: FloatArray | None = None,
) -> GravityQuadraticStepResult:
    """Solve one global convex gravity displacement problem with OSQP."""
    variable_count = 3 * linearization.vertex_count
    if warm_start is not None:
        assert warm_start.shape == (linearization.vertex_count, 3)
        assert warm_start.dtype == np.float64
    solver = osqp.OSQP()
    solver.setup(
        P=proximal_hessian(
            linearization.vertex_count, configuration.proximal_step
        ),
        q=gravity_linear_cost(linearization.vertex_count),
        A=linearization.matrix,
        l=linearization.lower_bounds,
        u=linearization.upper_bounds,
        eps_abs=configuration.absolute_tolerance,
        eps_rel=configuration.relative_tolerance,
        max_iter=configuration.maximum_iterations,
        polishing=configuration.polishing,
        verbose=False,
        warm_starting=True,
    )
    if warm_start is not None:
        solver.warm_start(x=warm_start.reshape(variable_count))
    solution = solver.solve(raise_error=False)
    status = str(solution.info.status).lower()
    succeeded = status == "solved" and solution.x is not None
    displacement = (
        np.asarray(solution.x, dtype=np.float64).reshape((-1, 3))
        if succeeded
        else None
    )
    return GravityQuadraticStepResult(
        displacement=displacement,
        status=status,
        succeeded=succeeded,
        objective=float(solution.info.obj_val),
        primal_residual=float(solution.info.prim_res),
        dual_residual=float(solution.info.dual_res),
        setup_time=float(solution.info.setup_time),
        solve_time=float(solution.info.solve_time),
        iterations=int(solution.info.iter),
        warm_started=warm_start is not None,
    )


def solve_minimum_norm_correction(
    matrix: csc_matrix,
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    *,
    vertex_count: int,
    maximum_coordinate_correction: float,
    tolerance: float = 1e-7,
    maximum_iterations: int = 4_000,
) -> GravityQuadraticStepResult:
    """Solve a bounded sparse minimum-norm correction without contact fill."""
    assert matrix.shape[1] == 3 * vertex_count
    assert lower_bounds.shape == upper_bounds.shape == (matrix.shape[0],)
    assert maximum_coordinate_correction > 0.0
    trust = eye(3 * vertex_count, format="csc", dtype=np.float64)
    complete_matrix = csc_matrix(vstack((matrix, trust), format="csc"))
    complete_lower = np.concatenate(
        (
            lower_bounds,
            np.full(3 * vertex_count, -maximum_coordinate_correction),
        )
    )
    complete_upper = np.concatenate(
        (
            upper_bounds,
            np.full(3 * vertex_count, maximum_coordinate_correction),
        )
    )
    solver = osqp.OSQP()
    solver.setup(
        P=csc_matrix(eye(3 * vertex_count, format="csc", dtype=np.float64)),
        q=np.zeros(3 * vertex_count, dtype=np.float64),
        A=complete_matrix,
        l=complete_lower,
        u=complete_upper,
        eps_abs=tolerance,
        eps_rel=tolerance,
        max_iter=maximum_iterations,
        polishing=True,
        verbose=False,
        warm_starting=True,
    )
    solution = solver.solve(raise_error=False)
    status = str(solution.info.status).lower()
    succeeded = status == "solved" and solution.x is not None
    displacement = (
        np.asarray(solution.x, dtype=np.float64).reshape((-1, 3))
        if succeeded
        else None
    )
    return GravityQuadraticStepResult(
        displacement=displacement,
        status=status,
        succeeded=succeeded,
        objective=float(solution.info.obj_val),
        primal_residual=float(solution.info.prim_res),
        dual_residual=float(solution.info.dual_res),
        setup_time=float(solution.info.setup_time),
        solve_time=float(solution.info.solve_time),
        iterations=int(solution.info.iter),
        warm_started=False,
    )
