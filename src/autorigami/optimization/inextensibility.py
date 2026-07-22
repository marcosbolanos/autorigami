from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

FloatArray = npt.NDArray[np.float64]


def reference_edge_lengths(points: FloatArray) -> FloatArray:
    """Return the strictly positive length assigned to every chain edge."""
    _check_points(points)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.all(lengths > 0.0), "chain edges must have positive length"
    return np.asarray(lengths, dtype=np.float64)


def edge_constraint_residuals(points: FloatArray, lengths: FloatArray) -> FloatArray:
    """Evaluate c_i = 1/2 (|x_(i+1)-x_i|^2 - length_i^2)."""
    _check_chain(points, lengths)
    edges = np.diff(points, axis=0)
    return 0.5 * (np.einsum("ij,ij->i", edges, edges) - lengths * lengths)


def edge_constraint_jacobian(points: FloatArray, lengths: FloatArray) -> csr_matrix:
    """Build the sparse exact Jacobian of all per-edge equalities."""
    _check_chain(points, lengths)
    edge_count = len(points) - 1
    edges = np.diff(points, axis=0)
    rows = np.repeat(np.arange(edge_count), 6)
    columns = np.empty((edge_count, 6), dtype=np.int64)
    values = np.empty((edge_count, 6), dtype=np.float64)
    coordinates = np.arange(3)
    columns[:, :3] = 3 * np.arange(edge_count)[:, None] + coordinates
    columns[:, 3:] = 3 * (np.arange(edge_count) + 1)[:, None] + coordinates
    values[:, :3] = -edges
    values[:, 3:] = edges
    return csr_matrix(
        (values.reshape(-1), (rows, columns.reshape(-1))),
        shape=(edge_count, 3 * len(points)),
    )


def edge_constraint_hessian_product(
    multipliers: FloatArray,
    direction: FloatArray,
) -> FloatArray:
    """Apply the exact Hessian of sum_i multipliers_i c_i."""
    _check_points(direction)
    assert multipliers.shape == (len(direction) - 1,)
    edge_direction = np.diff(direction, axis=0)
    weighted = multipliers[:, None] * edge_direction
    product = np.zeros_like(direction)
    product[:-1] -= weighted
    product[1:] += weighted
    return product


def maximum_edge_length_error(points: FloatArray, lengths: FloatArray) -> float:
    """Return max_i abs(|edge_i| - length_i), in coordinate units."""
    _check_chain(points, lengths)
    return float(np.max(np.abs(np.linalg.norm(np.diff(points, axis=0), axis=1) - lengths)))


def _check_points(points: FloatArray) -> None:
    assert points.dtype == np.float64
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) >= 2
    assert np.all(np.isfinite(points))


def _check_chain(points: FloatArray, lengths: FloatArray) -> None:
    _check_points(points)
    assert lengths.dtype == np.float64
    assert lengths.shape == (len(points) - 1,)
    assert np.all(lengths > 0.0)
