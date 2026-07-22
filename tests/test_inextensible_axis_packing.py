from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from autorigami._native import segment_segment_distance_parameters
from autorigami.optimization.axis_packing import optimize_axis_packing
from autorigami.optimization.constraints import ActiveConstraintJacobian
from autorigami.optimization.geometric_hessian import active_geometric_hessian_product
from autorigami.optimization.inextensibility import (
    edge_constraint_hessian_product,
    edge_constraint_jacobian,
    edge_constraint_residuals,
    maximum_edge_length_error,
    reference_edge_lengths,
)
from autorigami.optimization.nonlinear_projection import project_chain_lengths
from autorigami.optimization.nonlinear_projection import project_nonlinear_constraints
from autorigami.optimization.second_order import (
    axis_variance,
    axis_variance_gradient,
    lagrangian_hessian_product,
    lowest_tangent_mode,
    stationary_edge_multipliers,
)


def test_edge_derivatives_match_dense_finite_differences() -> None:
    rng = np.random.default_rng(42)
    points = np.cumsum(rng.normal(size=(8, 3)), axis=0).astype(np.float64)
    lengths = reference_edge_lengths(points)
    direction = rng.normal(size=points.shape).astype(np.float64)
    multipliers = rng.normal(size=len(lengths)).astype(np.float64)
    jacobian = edge_constraint_jacobian(points, lengths)
    epsilon = 1e-6
    finite_difference = (
        edge_constraint_residuals(points + epsilon * direction, lengths)
        - edge_constraint_residuals(points - epsilon * direction, lengths)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(jacobian @ direction.reshape(-1), finite_difference, atol=1e-8)

    def gradient(candidate: np.ndarray) -> np.ndarray:
        return np.asarray(
            edge_constraint_jacobian(candidate, lengths).T @ multipliers
        ).reshape((-1, 3))

    finite_hessian = (gradient(points + epsilon * direction) - gradient(points - epsilon * direction)) / (2.0 * epsilon)
    analytic_hessian = edge_constraint_hessian_product(multipliers, direction)
    np.testing.assert_allclose(analytic_hessian, finite_hessian, atol=1e-8)
    translation = np.broadcast_to(np.array([1.0, -2.0, 3.0]), points.shape).copy()
    np.testing.assert_allclose(edge_constraint_hessian_product(multipliers, translation), 0.0)
    other = rng.normal(size=points.shape)
    left = float(other.reshape(-1) @ analytic_hessian.reshape(-1))
    right = float(direction.reshape(-1) @ edge_constraint_hessian_product(multipliers, other).reshape(-1))
    assert abs(left - right) < 1e-10


def test_straight_chain_escapes_through_negative_reduced_hessian_mode() -> None:
    points = np.zeros((21, 3), dtype=np.float64)
    points[:, 2] = 0.34 * np.arange(len(points))
    lengths = reference_edge_lengths(points)
    gradient = axis_variance_gradient(points, 2)
    jacobian = edge_constraint_jacobian(points, lengths)
    multipliers = stationary_edge_multipliers(points, lengths, gradient)
    assert np.linalg.norm(gradient.reshape(-1) + jacobian.T @ multipliers) < 1e-10

    def hessian(direction: np.ndarray) -> np.ndarray:
        return lagrangian_hessian_product(
            direction,
            axis=2,
            edge_multipliers=multipliers,
        )

    mode = lowest_tangent_mode(points, lengths, hessian)
    assert mode.eigenvalue < -1e-6
    assert mode.tangent_residual < 1e-8
    projected = project_chain_lengths(points + 0.1 * mode.displacement, lengths)
    assert projected.converged
    assert maximum_edge_length_error(projected.points, lengths) < 1e-9
    assert axis_variance(projected.points, 2) < axis_variance(points, 2)
    assert np.max(np.linalg.norm(projected.points[:, :2], axis=1)) > 1e-3

    workload = optimize_axis_packing(
        points.astype(np.float32),
        axis="z",
        maximum_angle=0.2,
        minimum_distance=0.1,
        ignored_adjacent_edges=100,
        maximum_iterations=1,
        maximum_vertex_step=0.1,
    )
    assert workload.iterations[0].used_negative_curvature_mode
    assert workload.final_objective < workload.initial_objective
    assert np.ptp(workload.points[:, :2]) > 1e-3


def test_active_contact_hessian_is_symmetric_and_translation_invariant() -> None:
    points = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 2.7], [0.0, 1.0, 2.7]],
        dtype=np.float64,
    )
    parameters = np.asarray(
        segment_segment_distance_parameters(
            points.astype(np.float32), [(0, 2)]
        ),
        dtype=np.float64,
    )
    constraints = ActiveConstraintJacobian(
        matrix=csr_matrix((1, points.size), dtype=np.float64),
        slacks=np.array([0.1]),
        vertex_count=4,
        contact_count=1,
        curvature_count=0,
        contact_pairs=((0, 2),),
        contact_parameters=parameters,
    )
    multipliers = np.array([0.7])
    basis_products = []
    for column in range(points.size):
        direction = np.zeros_like(points)
        direction.reshape(-1)[column] = 1.0
        basis_products.append(
            active_geometric_hessian_product(points, constraints, multipliers, direction).reshape(-1)
        )
    dense = np.column_stack(basis_products)
    np.testing.assert_allclose(dense, dense.T, atol=1e-10)
    translation = np.broadcast_to(np.array([0.4, -0.3, 0.2]), points.shape).copy()
    np.testing.assert_allclose(
        active_geometric_hessian_product(points, constraints, multipliers, translation),
        0.0,
        atol=1e-10,
    )


def test_active_curvature_hessian_is_symmetric() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.9, 0.3, 0.1]],
        dtype=np.float64,
    )
    constraints = ActiveConstraintJacobian(
        matrix=csr_matrix((1, points.size), dtype=np.float64),
        slacks=np.array([0.01]),
        vertex_count=3,
        contact_count=0,
        curvature_count=1,
        curvature_vertices=np.array([1], dtype=np.int64),
    )
    multiplier = np.array([1.3])
    dense = np.column_stack(
        [
            active_geometric_hessian_product(
                points,
                constraints,
                multiplier,
                np.eye(points.size, dtype=np.float64)[:, column].reshape((-1, 3)),
            ).reshape(-1)
            for column in range(points.size)
        ]
    )
    np.testing.assert_allclose(dense, dense.T, atol=1e-9)


def test_combined_projection_satisfies_edges_and_a_nonlinear_inequality() -> None:
    points = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    lengths = reference_edge_lengths(points)

    def inequality_factory(candidate: np.ndarray) -> ActiveConstraintJacobian:
        matrix = np.zeros((1, candidate.size), dtype=np.float64)
        matrix[0, 4] = 1.0
        return ActiveConstraintJacobian(
            matrix=csr_matrix(matrix),
            slacks=np.array([candidate[1, 1] - 0.1]),
            vertex_count=3,
            contact_count=0,
            curvature_count=1,
        )

    projection = project_nonlinear_constraints(points, lengths, inequality_factory)
    assert projection.converged
    assert projection.minimum_slack >= -1e-6
    assert projection.edge_error <= 1e-6
    assert projection.barycenter_error <= 1e-8
