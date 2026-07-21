from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator, cg

from autorigami.optimization.constraints import ActiveConstraintJacobian
from autorigami.optimization.fractional_sobolev import (
    FractionalSobolevPreconditioner,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ProjectedDirection:
    """Fractional-metric descent direction and KKT solve diagnostics."""

    displacement: FloatArray
    iterations: int
    residual: float


def solve_fractional_kkt(
    differential: FloatArray,
    *,
    metric: FractionalSobolevPreconditioner,
    constraints: ActiveConstraintJacobian,
    regularization: float,
    relative_tolerance: float,
    maximum_iterations: int,
) -> ProjectedDirection:
    """Project a gradient into active constraints in the Sobolev metric.

    The Schur complement ``J M^-1 J.T`` is applied matrix-free. Each Krylov
    iteration therefore needs one sparse constraint scatter/gather and one
    cosine-transform fractional inverse; no dense vertex or KKT matrix is
    assembled.
    """
    assert differential.shape == (metric.vertex_count, 3)
    assert constraints.vertex_count == metric.vertex_count
    assert regularization > 0.0
    assert relative_tolerance > 0.0
    assert maximum_iterations > 0

    unconstrained = metric.apply_inverse(differential)
    if constraints.constraint_count == 0:
        return ProjectedDirection(
            displacement=-unconstrained,
            iterations=0,
            residual=0.0,
        )

    right_hand_side = constraints.apply(unconstrained)
    iteration_count = 0

    def schur_product(values: npt.ArrayLike) -> FloatArray:
        multipliers = np.asarray(values, dtype=np.float64)
        scattered = constraints.transpose_apply(multipliers)
        projected = metric.apply_inverse(scattered)
        return constraints.apply(projected) + regularization * multipliers

    def count_iteration(_: FloatArray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    operator = LinearOperator(
        shape=(constraints.constraint_count, constraints.constraint_count),
        matvec=schur_product,  # pyright: ignore[reportCallIssue]
        dtype=np.float64,
    )
    multipliers, info = cg(
        operator,
        right_hand_side,
        rtol=relative_tolerance,
        atol=0.0,
        maxiter=maximum_iterations,
        callback=count_iteration,
    )
    if info != 0:
        raise RuntimeError(
            "fractional KKT solve did not converge within "
            f"{maximum_iterations} iterations"
        )
    corrected_differential = differential - constraints.transpose_apply(multipliers)
    displacement = -metric.apply_inverse(corrected_differential)
    residual = float(
        np.linalg.norm(schur_product(multipliers) - right_hand_side)
        / max(1.0, np.linalg.norm(right_hand_side))
    )
    return ProjectedDirection(
        displacement=displacement,
        iterations=iteration_count,
        residual=residual,
    )
