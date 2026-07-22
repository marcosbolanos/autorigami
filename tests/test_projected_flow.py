import numpy as np
from scipy.sparse import csr_matrix

from autorigami._native import segment_segment_distance
from autorigami.geometry.curvature import get_polyline_angles
from autorigami.optimization.constraints import (
    ActiveConstraintJacobian,
    active_constraint_jacobian,
)
from autorigami.optimization.fractional_sobolev import (
    FractionalSobolevPreconditioner,
)
from autorigami.optimization.projected_flow import solve_fractional_kkt


def test_matrix_free_fractional_kkt_matches_dense_reference() -> None:
    points = np.column_stack(
        (
            np.linspace(0.0, 2.0, 7),
            0.1 * np.sin(np.linspace(0.0, np.pi, 7)),
            np.zeros(7),
        )
    )
    metric = FractionalSobolevPreconditioner(points)
    rng = np.random.default_rng(4)
    differential = rng.normal(size=points.shape)
    dense_jacobian = rng.normal(size=(4, points.size))
    constraints = ActiveConstraintJacobian(
        matrix=csr_matrix(dense_jacobian),
        slacks=np.ones(4),
        vertex_count=len(points),
        contact_count=2,
        curvature_count=2,
    )
    regularization = 1e-7

    result = solve_fractional_kkt(
        differential,
        metric=metric,
        constraints=constraints,
        required_constraint_changes=np.zeros(4),
        regularization=regularization,
        relative_tolerance=1e-11,
        maximum_iterations=100,
    )

    inverse_metric = np.empty((points.size, points.size))
    for column in range(points.size):
        basis = np.zeros(points.size)
        basis[column] = 1.0
        inverse_metric[:, column] = metric.apply_inverse(
            basis.reshape(points.shape)
        ).reshape(-1)
    flat_differential = differential.reshape(-1)
    schur = dense_jacobian @ inverse_metric @ dense_jacobian.T
    multipliers = np.linalg.solve(
        schur + regularization * np.eye(len(schur)),
        dense_jacobian @ inverse_metric @ flat_differential,
    )
    expected = -inverse_metric @ (flat_differential - dense_jacobian.T @ multipliers)

    assert np.allclose(result.displacement.reshape(-1), expected, rtol=2e-6, atol=1e-6)
    assert result.residual < 1e-9


def test_active_curvature_row_matches_finite_difference() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.8, 0.6, 0.0]],
        dtype=np.float64,
    )
    angle = float(get_polyline_angles(points.astype(np.float32))[0])
    constraints = active_constraint_jacobian(
        points,
        target_angle=angle + 0.01,
        min_distance=0.1,
        ignored_adjacent_edges=1,
        contact_activation_distance=0.0,
        curvature_activation_angle=0.02,
    )
    displacement = np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1], [0.3, 0.1, -0.2]])
    epsilon = 1e-5
    plus_angle = float(
        get_polyline_angles((points + epsilon * displacement).astype(np.float32))[0]
    )
    minus_angle = float(
        get_polyline_angles((points - epsilon * displacement).astype(np.float32))[0]
    )
    finite_difference = -(plus_angle - minus_angle) / (2.0 * epsilon)

    assert constraints.contact_count == 0
    assert constraints.curvature_count == 1
    assert np.isclose(
        constraints.apply(displacement)[0],
        finite_difference,
        rtol=2e-2,
        atol=2e-3,
    )


def test_active_contact_row_matches_finite_difference() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    constraints = active_constraint_jacobian(
        points,
        target_angle=3.0,
        min_distance=1.9,
        ignored_adjacent_edges=2,
        contact_activation_distance=0.2,
        curvature_activation_angle=0.0,
    )
    displacement = np.array(
        [
            [0.1, -0.2, 0.0],
            [-0.3, 0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.3, 0.0],
            [-0.1, -0.2, 0.0],
        ]
    )
    epsilon = 1e-4

    def contact_distance(candidate: np.ndarray) -> float:
        return float(
            segment_segment_distance(np.asarray(candidate, dtype=np.float32), [(0, 3)])[
                0
            ][0]
        )

    finite_difference = (
        contact_distance(points + epsilon * displacement)
        - contact_distance(points - epsilon * displacement)
    ) / (2.0 * epsilon)

    assert constraints.contact_count == 1
    assert constraints.curvature_count == 0
    assert np.isclose(
        constraints.apply(displacement)[0],
        finite_difference,
        rtol=2e-3,
        atol=2e-3,
    )
