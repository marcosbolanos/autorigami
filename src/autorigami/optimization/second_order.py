from __future__ import annotations

# pyright: reportOptionalSubscript=false, reportCallIssue=false, reportArgumentType=false

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import LinearOperator, eigsh, lsmr

from autorigami.optimization.inextensibility import (
    edge_constraint_hessian_product,
    edge_constraint_jacobian,
)

FloatArray = npt.NDArray[np.float64]
HessianProduct = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class ReducedHessianMode:
    """Lowest eigenpair of the Lagrangian Hessian in the feasible tangent space."""

    eigenvalue: float
    displacement: FloatArray
    tangent_residual: float

    @property
    def is_negative(self) -> bool:
        return self.eigenvalue < -1e-10


def axis_variance(points: FloatArray, axis: int) -> float:
    """Mean squared distance from the centroid along one coordinate."""
    _check_axis_inputs(points, axis)
    centered = points[:, axis] - np.mean(points[:, axis])
    return 0.5 * float(centered @ centered) / len(points)


def axis_variance_gradient(points: FloatArray, axis: int) -> FloatArray:
    """Exact differential of :func:`axis_variance`."""
    _check_axis_inputs(points, axis)
    gradient = np.zeros_like(points)
    gradient[:, axis] = (points[:, axis] - np.mean(points[:, axis])) / len(points)
    return gradient


def axis_variance_hessian_product(direction: FloatArray, axis: int) -> FloatArray:
    """Apply the constant exact Hessian of mean axis variance."""
    _check_axis_inputs(direction, axis)
    product = np.zeros_like(direction)
    product[:, axis] = (direction[:, axis] - np.mean(direction[:, axis])) / len(direction)
    return product


def stationary_edge_multipliers(
    points: FloatArray,
    lengths: FloatArray,
    objective_gradient: FloatArray,
) -> FloatArray:
    """Least-squares multipliers satisfying grad(f) + J.T lambda = 0."""
    assert objective_gradient.shape == points.shape
    jacobian = edge_constraint_jacobian(points, lengths)
    solution = lsmr(
        jacobian.T,
        -objective_gradient.reshape(-1),
        atol=1e-12,
        btol=1e-12,
        maxiter=max(100, 4 * len(lengths)),
    )
    if solution[1] not in (1, 2):  # pyright: ignore[reportOptionalSubscript]
        raise RuntimeError(  # pyright: ignore[reportOptionalSubscript]
            f"edge multiplier solve failed with LSMR status {solution[1]}"
        )
    return np.asarray(solution[0], dtype=np.float64)  # pyright: ignore[reportOptionalSubscript]


def lagrangian_hessian_product(
    direction: FloatArray,
    *,
    axis: int,
    edge_multipliers: FloatArray,
    geometric_hessian_product: HessianProduct | None = None,
) -> FloatArray:
    """Apply objective, edge, and optional active-geometric Hessian terms."""
    product = axis_variance_hessian_product(direction, axis)
    product += edge_constraint_hessian_product(edge_multipliers, direction)
    if geometric_hessian_product is not None:
        geometric = geometric_hessian_product(direction)
        assert geometric.shape == direction.shape
        product += geometric
    return product


def lowest_tangent_mode(
    points: FloatArray,
    lengths: FloatArray,
    hessian_product: HessianProduct,
    *,
    tolerance: float = 1e-8,
    maximum_iterations: int = 500,
) -> ReducedHessianMode:
    """Find the lowest constrained Hessian mode with rigid motions removed."""
    assert points.dtype == np.float64 and lengths.dtype == np.float64
    tangent = _tangent_and_rigid_rows(points, lengths)
    size = points.size

    def project(flat: npt.ArrayLike) -> FloatArray:
        vector = np.asarray(flat, dtype=np.float64)
        coefficients = lsmr(
            tangent.T,
            vector,
            atol=1e-11,
            btol=1e-11,
            maxiter=max(100, 4 * tangent.shape[0]),
        )[0]  # pyright: ignore[reportOptionalSubscript]
        return vector - np.asarray(tangent.T @ coefficients).reshape(-1)

    def product(flat: npt.ArrayLike) -> FloatArray:
        feasible = project(flat)
        applied = hessian_product(feasible.reshape((-1, 3))).reshape(-1)
        return project(applied)

    operator = LinearOperator(  # pyright: ignore[reportCallIssue]
        shape=(size, size), matvec=product, dtype=np.float64
    )
    values, vectors = eigsh(  # pyright: ignore[reportCallIssue, reportArgumentType]
        operator,
        k=1,
        which="SA",
        tol=tolerance,
        maxiter=maximum_iterations,
    )
    displacement = project(vectors[:, 0]).reshape((-1, 3))
    displacement /= np.max(np.linalg.norm(displacement, axis=1))
    residual = float(np.linalg.norm(tangent @ displacement.reshape(-1)))
    return ReducedHessianMode(float(values[0]), displacement, residual)


def _tangent_and_rigid_rows(points: FloatArray, lengths: FloatArray) -> csr_matrix:
    edge_rows = edge_constraint_jacobian(points, lengths)
    centered = points - np.mean(points, axis=0)
    rigid: list[FloatArray] = []
    for coordinate in range(3):
        translation = np.zeros_like(points)
        translation[:, coordinate] = 1.0
        rigid.append(translation.reshape(-1))
    for axis in np.eye(3):
        rotation = np.asarray(
            np.cross(np.broadcast_to(axis, centered.shape), centered),
            dtype=np.float64,
        )
        rigid.append(rotation.reshape(-1))
    independent: list[FloatArray] = []
    for row in rigid:
        candidate = row.copy()
        for basis in independent:
            candidate -= (candidate @ basis) * basis
        norm = np.linalg.norm(candidate)
        if norm > 1e-10:
            independent.append(candidate / norm)
    rigid_rows = csr_matrix(np.vstack(independent))
    return csr_matrix(vstack((edge_rows, rigid_rows), format="csr"))


def _check_axis_inputs(points: FloatArray, axis: int) -> None:
    assert points.dtype == np.float64
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2
    assert axis in (0, 1, 2)
