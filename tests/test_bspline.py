import numpy as np

from src.curvepack.bspline import (
    bspline_basis_one,
    make_basis_matrix,
    make_clamped_uniform_knots,
)


def test_make_clamped_uniform_knots_cubic() -> None:
    knots = make_clamped_uniform_knots(n_ctrl=4, degree=3)
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(knots, expected)


def test_bspline_basis_one_linear() -> None:
    knots = make_clamped_uniform_knots(n_ctrl=2, degree=1)

    assert np.isclose(bspline_basis_one(0.0, 0, 1, knots), 1.0)
    assert np.isclose(bspline_basis_one(0.0, 1, 1, knots), 0.0)

    assert np.isclose(bspline_basis_one(0.5, 0, 1, knots), 0.5)
    assert np.isclose(bspline_basis_one(0.5, 1, 1, knots), 0.5)

    assert np.isclose(bspline_basis_one(1.0, 0, 1, knots), 0.0)
    assert np.isclose(bspline_basis_one(1.0, 1, 1, knots), 1.0)


def test_make_basis_matrix_linear_interpolation() -> None:
    B = make_basis_matrix(n_ctrl=2, n_samples=3, degree=1)
    expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(B, expected, atol=1e-6)


def test_make_basis_matrix_partition_unity_cubic() -> None:
    B = make_basis_matrix(n_ctrl=6, n_samples=11, degree=3)
    row_sum = B.sum(axis=1)

    np.testing.assert_allclose(row_sum, np.ones_like(row_sum), atol=1e-6)
    assert float(B.min()) >= -1e-6
    assert np.isclose(B[0, 0], 1.0)
    assert np.isclose(B[-1, -1], 1.0)
