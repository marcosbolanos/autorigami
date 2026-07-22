from __future__ import annotations

# pyright: reportOptionalSubscript=false

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import LinearOperator, cg

from autorigami.optimization.constraints import (
    ActiveConstraintJacobian,
    EqualityConstraintSystem,
)

FloatArray = npt.NDArray[np.float64]


class InverseMetric(Protocol):
    """Metric inverse required by the matrix-free projected solvers."""

    @property
    def vertex_count(self) -> int: ...

    def apply_inverse(self, differential: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class ProjectedDirection:
    """Fractional-metric descent direction and KKT solve diagnostics."""

    displacement: FloatArray
    iterations: int
    residual: float
    multipliers: FloatArray
    active_constraint_indices: npt.NDArray[np.int64]


def solve_fractional_kkt(
    differential: FloatArray,
    *,
    metric: InverseMetric,
    constraints: ActiveConstraintJacobian,
    regularization: float,
    relative_tolerance: float,
    maximum_iterations: int,
    required_constraint_changes: FloatArray | None = None,
) -> ProjectedDirection:
    """Project a gradient into active constraints in the Sobolev metric.

    The Schur complement ``J M^-1 J.T`` is applied matrix-free. Each Krylov
    iteration therefore needs one sparse constraint scatter/gather and one
    cosine-transform fractional inverse; no dense vertex or KKT matrix is
    assembled.
    """
    assert differential.shape == (metric.vertex_count, 3)
    assert constraints.vertex_count == metric.vertex_count
    if required_constraint_changes is None:
        required_constraint_changes = np.zeros(
            constraints.constraint_count, dtype=np.float64
        )
    assert required_constraint_changes.shape == (constraints.constraint_count,)
    assert regularization > 0.0
    assert relative_tolerance > 0.0
    assert maximum_iterations > 0

    unconstrained = metric.apply_inverse(differential)
    if constraints.constraint_count == 0:
        return ProjectedDirection(
            displacement=-unconstrained,
            iterations=0,
            residual=0.0,
            multipliers=np.empty(0, dtype=np.float64),
            active_constraint_indices=np.empty(0, dtype=np.int64),
        )

    right_hand_side = constraints.apply(unconstrained) + required_constraint_changes
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
        multipliers=np.asarray(multipliers, dtype=np.float64),
        active_constraint_indices=np.arange(
            constraints.constraint_count, dtype=np.int64
        ),
    )


def solve_fractional_mixed_step(
    differential: FloatArray,
    *,
    metric: InverseMetric,
    equalities: EqualityConstraintSystem,
    inequalities: ActiveConstraintJacobian,
    maximum_vertex_step: float,
    safety_margin: float,
    regularization: float,
    relative_tolerance: float,
    maximum_iterations: int,
    maximum_active_set_iterations: int = 50,
) -> ProjectedDirection:
    """Solve one metric step with permanent equalities and blocking bounds."""
    assert equalities.vertex_count == metric.vertex_count == inequalities.vertex_count
    unconstrained = -metric.apply_inverse(differential)
    largest = float(np.max(np.linalg.norm(unconstrained, axis=1), initial=0.0))
    if largest <= 1e-14:
        return ProjectedDirection(
            np.zeros_like(differential), 0, 0.0, np.empty(0), np.empty(0, dtype=np.int64)
        )
    scaled = differential * min(1.0, maximum_vertex_step / largest)
    active = np.zeros(inequalities.constraint_count, dtype=np.bool_)
    total_iterations = 0
    for _ in range(maximum_active_set_iterations):
        indices = np.flatnonzero(active).astype(np.int64)
        selected = inequalities.select(indices)
        matrix = csr_matrix(vstack((equalities.jacobian, selected.matrix), format="csr"))
        combined = ActiveConstraintJacobian(
            matrix=matrix,
            slacks=np.concatenate((equalities.values, selected.slacks)),
            vertex_count=metric.vertex_count,
            contact_count=matrix.shape[0],
            curvature_count=0,
        )
        required = np.concatenate(
            (-equalities.values, safety_margin - selected.slacks)
        )
        projected = solve_fractional_kkt(
            scaled,
            metric=metric,
            constraints=combined,
            required_constraint_changes=required,
            regularization=regularization,
            relative_tolerance=relative_tolerance,
            maximum_iterations=maximum_iterations,
        )
        total_iterations += projected.iterations
        inequality_multipliers = projected.multipliers[len(equalities.values) :]
        if len(inequality_multipliers) and np.min(inequality_multipliers) < -1e-10:
            active[indices[int(np.argmin(inequality_multipliers))]] = False
            continue
        predicted = inequalities.slacks + inequalities.apply(projected.displacement)
        blocking = (~active) & (predicted < safety_margin)
        if np.any(blocking):
            candidates = np.flatnonzero(blocking)
            active[candidates[int(np.argmin(predicted[candidates]))]] = True
            continue
        displacement = projected.displacement
        step = float(np.max(np.linalg.norm(displacement, axis=1), initial=0.0))
        if step > maximum_vertex_step:
            displacement *= maximum_vertex_step / step
        return ProjectedDirection(
            displacement=displacement,
            iterations=total_iterations,
            residual=projected.residual,
            multipliers=projected.multipliers,
            active_constraint_indices=indices,
        )
    raise RuntimeError("mixed equality/inequality active set did not converge")
