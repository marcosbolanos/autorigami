from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def make_clamped_uniform_knots(n_ctrl: int, degree: int = 3) -> Float[np.ndarray, "m"]:
    """
    Clamped uniform knot vector for n_ctrl control points and given degree.
    Knot count = n_ctrl + degree + 1
    """
    p = degree
    m = n_ctrl + p + 1
    # Uniform internal knots from 0..1
    knots = np.zeros(m, dtype=np.float32)
    knots[p : m - p] = np.linspace(0.0, 1.0, m - 2 * p, dtype=np.float32)
    knots[m - p :] = 1.0
    return knots


@jaxtyped(typechecker=beartype)
def bspline_basis_one(
    t: float,
    i: int,
    p: int,
    knots: Float[np.ndarray, "m"],
) -> float:
    """
    Cox-de Boor recursion for basis function N_{i,p}(t).
    Scalar version for building basis matrix offline.
    """
    if p == 0:
        return (
            1.0
            if (
                knots[i] <= t < knots[i + 1]
                or (t == 1.0 and knots[i] <= t <= knots[i + 1])
            )
            else 0.0
        )
    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]
    a = 0.0
    b = 0.0
    if denom1 > 0:
        a = (t - knots[i]) / denom1 * bspline_basis_one(t, i, p - 1, knots)
    if denom2 > 0:
        b = (knots[i + p + 1] - t) / denom2 * bspline_basis_one(t, i + 1, p - 1, knots)
    return float(a + b)


@jaxtyped(typechecker=beartype)
def make_basis_matrix(
    n_ctrl: int,
    n_samples: int,
    degree: int = 3,
) -> Float[np.ndarray, "n_samples n_ctrl"]:
    """
    Returns B: (n_samples, n_ctrl) such that X = B @ P for one curve.
    """
    knots = make_clamped_uniform_knots(n_ctrl, degree)
    ts = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    B_mat = np.zeros((n_samples, n_ctrl), dtype=np.float32)
    for k, t in enumerate(ts):
        for i in range(n_ctrl):
            B_mat[k, i] = bspline_basis_one(float(t), i, degree, knots)
    # Normalize rows for numerical stability (should already sum to 1)
    row_sum = B_mat.sum(axis=1, keepdims=True)
    B_mat = B_mat / np.maximum(row_sum, 1e-8)
    return B_mat
