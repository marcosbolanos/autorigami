from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csc_matrix, eye

from autorigami.gravity.constraints import (
    GravityConstraintConfiguration,
    GravityConstraintLinearization,
    GravityConstraintResiduals,
    contact_candidate_radius,
    floor_constraint_jacobian,
    floor_slacks,
    linearize_gravity_constraints,
)
from autorigami.gravity.projection import (
    GravityProjectionConfiguration,
    GravityProjectionResult,
    project_gravity_constraints,
)
from autorigami.gravity.quadratic_step import (
    GravityQuadraticStepConfiguration,
    gravity_linear_cost,
    proximal_hessian,
    solve_gravity_quadratic_step,
)
from autorigami.gravity.quasistatic import (
    QuasistaticGravityConfiguration,
    gravitational_energy,
    minimize_gravitational_energy,
)
from autorigami.gravity.workloads import pack_under_gravity
from autorigami.geometry.validation import PolylineValidationResult
from autorigami.optimization.inextensibility import reference_edge_lengths


def _horizontal_chain(vertex_count: int = 30) -> np.ndarray:
    points = np.zeros((vertex_count, 3), dtype=np.float32)
    points[:, 0] = 0.34 * np.arange(vertex_count)
    points[:, 2] = 1.0
    return points


def test_floor_slack_jacobian_matches_finite_differences() -> None:
    rng = np.random.default_rng(42)
    points = rng.normal(size=(7, 3)).astype(np.float64)
    direction = rng.normal(size=points.shape).astype(np.float64)
    matrix = floor_constraint_jacobian(len(points))
    epsilon = 1e-6
    finite_difference = (
        floor_slacks(
            points + epsilon * direction,
            floor_height=-2.0,
            tube_radius=1.3,
        )
        - floor_slacks(
            points - epsilon * direction,
            floor_height=-2.0,
            tube_radius=1.3,
        )
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        matrix @ direction.reshape(-1), finite_difference, atol=1e-9
    )


def test_constraint_assembly_contains_every_documented_row_group() -> None:
    points = np.asarray(_horizontal_chain(8), dtype=np.float64)
    lengths = reference_edge_lengths(points)
    configuration = GravityConstraintConfiguration()
    linearization = linearize_gravity_constraints(
        points,
        lengths,
        floor_height=-0.301,
        tube_radius=1.3,
        trust_radius=0.05,
        configuration=configuration,
    )
    assert linearization.edge_count == 7
    assert linearization.curvature_count == 6
    assert linearization.contact_count == 0
    assert linearization.floor_count == 8
    assert linearization.trust_count == 24
    assert linearization.matrix.shape == (45, 24)

    edge_residual = np.asarray(
        linearization.matrix[:7] @ np.zeros(24), dtype=np.float64
    )
    np.testing.assert_array_equal(edge_residual, 0.0)
    np.testing.assert_array_equal(
        linearization.lower_bounds[:7], linearization.upper_bounds[:7]
    )
    geometric_start = linearization.edge_count
    geometric_end = geometric_start + 6 + 0 + 8
    assert np.all(np.isposinf(linearization.upper_bounds[geometric_start:geometric_end]))
    trust_bound = 0.05 / np.sqrt(3.0)
    np.testing.assert_allclose(
        linearization.lower_bounds[-24:], -trust_bound
    )
    np.testing.assert_allclose(
        linearization.upper_bounds[-24:], trust_bound
    )


def test_edge_equalities_include_current_nonlinear_residual() -> None:
    points = np.asarray(_horizontal_chain(6), dtype=np.float64)
    lengths = reference_edge_lengths(points)
    points[3, 0] += 0.02
    linearization = linearize_gravity_constraints(
        points,
        lengths,
        floor_height=-0.301,
        tube_radius=1.3,
        trust_radius=0.05,
        configuration=GravityConstraintConfiguration(),
    )
    assert np.any(np.abs(linearization.lower_bounds[:5]) > 1e-6)
    np.testing.assert_array_equal(
        linearization.lower_bounds[:5], linearization.upper_bounds[:5]
    )


def test_contact_search_radius_covers_two_bounded_segment_movements() -> None:
    configuration = GravityConstraintConfiguration(contact_safety_margin=0.01)
    trust_radius = 0.1
    assert contact_candidate_radius(configuration, trust_radius) == 2.81
    points = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 4.0],
            [-1.0, 0.0, 2.79],
            [1.0, 0.0, 2.79],
        ],
        dtype=np.float64,
    )
    lengths = reference_edge_lengths(points)
    linearization = linearize_gravity_constraints(
        points,
        lengths,
        floor_height=-10.0,
        tube_radius=1.3,
        trust_radius=trust_radius,
        configuration=replace(configuration, ignored_adjacent_edges=1),
    )
    assert (0, 3) in linearization.contact_pairs


def test_proximal_qp_has_exact_unconstrained_downward_step() -> None:
    vertex_count = 4
    trust_bound = 1.0
    linearization = GravityConstraintLinearization(
        matrix=csc_matrix(eye(3 * vertex_count, format="csc")),
        lower_bounds=np.full(3 * vertex_count, -trust_bound),
        upper_bounds=np.full(3 * vertex_count, trust_bound),
        vertex_count=vertex_count,
        edge_count=0,
        curvature_count=0,
        contact_count=0,
        floor_count=0,
        trust_count=3 * vertex_count,
        contact_pairs=(),
    )
    configuration = GravityQuadraticStepConfiguration(proximal_step=0.05)
    result = solve_gravity_quadratic_step(
        linearization,
        configuration=configuration,
    )
    assert result.succeeded
    assert result.displacement is not None
    expected = np.zeros((vertex_count, 3))
    expected[:, 2] = -0.05
    np.testing.assert_allclose(result.displacement, expected, atol=1e-7)
    np.testing.assert_array_equal(
        gravity_linear_cost(vertex_count).reshape((-1, 3))[:, 2], 1.0
    )
    np.testing.assert_allclose(
        proximal_hessian(vertex_count, 0.05).diagonal(), 20.0
    )


def test_one_sided_inequality_does_not_freeze_a_nonblocking_row() -> None:
    matrix = csc_matrix([[0.0, 0.0, 1.0], *np.eye(3)])
    linearization = GravityConstraintLinearization(
        matrix=matrix,
        lower_bounds=np.array([-0.1, -1.0, -1.0, -1.0]),
        upper_bounds=np.array([np.inf, 1.0, 1.0, 1.0]),
        vertex_count=1,
        edge_count=0,
        curvature_count=0,
        contact_count=1,
        floor_count=0,
        trust_count=3,
        contact_pairs=((0, 1),),
    )
    result = solve_gravity_quadratic_step(
        linearization,
        configuration=GravityQuadraticStepConfiguration(proximal_step=0.05),
    )
    assert result.displacement is not None
    assert abs(result.displacement[0, 2] + 0.05) < 1e-7
    assert result.displacement[0, 2] != -0.1


def test_blocking_contact_row_couples_the_vertices_without_crossing() -> None:
    contact_row = np.array([[1.0, 0.0, 0.0, -1.0, 0.0, 0.0]])
    matrix = csc_matrix(np.vstack((contact_row, np.eye(6))))
    linearization = GravityConstraintLinearization(
        matrix=matrix,
        lower_bounds=np.concatenate(([0.04], np.full(6, -0.1))),
        upper_bounds=np.concatenate(([np.inf], np.full(6, 0.1))),
        vertex_count=2,
        edge_count=0,
        curvature_count=0,
        contact_count=1,
        floor_count=0,
        trust_count=6,
        contact_pairs=((0, 1),),
    )
    result = solve_gravity_quadratic_step(
        linearization,
        configuration=GravityQuadraticStepConfiguration(proximal_step=0.05),
    )
    assert result.succeeded
    assert result.displacement is not None
    flattened = result.displacement.reshape(-1)
    assert float((contact_row @ flattened)[0]) >= 0.04 - 1e-7
    assert flattened[0] > 0.0
    assert flattened[3] < 0.0


def test_trust_rows_bound_each_vertex_euclidean_displacement() -> None:
    points = np.asarray(_horizontal_chain(12), dtype=np.float64)
    trust_radius = 0.03
    linearization = linearize_gravity_constraints(
        points,
        reference_edge_lengths(points),
        floor_height=-10.0,
        tube_radius=1.3,
        trust_radius=trust_radius,
        configuration=GravityConstraintConfiguration(),
    )
    result = solve_gravity_quadratic_step(
        linearization,
        configuration=GravityQuadraticStepConfiguration(proximal_step=0.1),
    )
    assert result.succeeded
    assert result.displacement is not None
    norms = np.linalg.norm(result.displacement, axis=1)
    assert float(np.max(norms)) <= trust_radius + 1e-8


def test_supported_vertex_stays_on_floor_while_chain_lowers_globally() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [0.34, 0.0, 0.34], [0.68, 0.0, 0.68]],
        dtype=np.float64,
    )
    configuration = GravityConstraintConfiguration(
        maximum_angle=3.13,
        ignored_adjacent_edges=1,
        curvature_safety_margin=0.0,
        contact_safety_margin=0.0,
    )
    linearization = linearize_gravity_constraints(
        points,
        reference_edge_lengths(points),
        floor_height=-1.3,
        tube_radius=1.3,
        trust_radius=0.05,
        configuration=configuration,
    )
    result = solve_gravity_quadratic_step(
        linearization,
        configuration=GravityQuadraticStepConfiguration(proximal_step=0.05),
    )
    assert result.succeeded
    assert result.displacement is not None
    assert result.displacement[0, 2] >= -1e-8
    assert result.displacement[-1, 2] < -0.02


def test_proximal_hessian_scales_sparsely_to_sixty_thousand_variables() -> None:
    hessian = proximal_hessian(20_000, 0.05)
    assert hessian.shape == (60_000, 60_000)
    assert hessian.nnz == 60_000
    assert hessian.dtype == np.float64
    np.testing.assert_array_equal(hessian.diagonal(), 20.0)


def test_gravity_projection_does_not_restore_barycenter() -> None:
    points = np.asarray(_horizontal_chain(10), dtype=np.float64)
    points[:, 2] = -0.1
    lengths = reference_edge_lengths(points)
    projected = project_gravity_constraints(
        points,
        lengths,
        floor_height=-1.3,
        tube_radius=1.3,
        constraint_configuration=GravityConstraintConfiguration(),
        projection_configuration=GravityProjectionConfiguration(),
    )
    assert projected.converged
    assert np.min(projected.points[:, 2]) >= -1e-6
    assert np.mean(projected.points[:, 2]) > np.mean(points[:, 2]) + 0.09


def test_projection_couples_edge_lengths_and_a_blocking_contact() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 3.0, 2.0],
            [1.0, 0.24, 0.0],
            [0.0, 0.24, 0.0],
        ],
        dtype=np.float64,
    )
    lengths = reference_edge_lengths(points)
    constraints = GravityConstraintConfiguration(
        maximum_angle=3.13,
        minimum_distance=0.25,
        ignored_adjacent_edges=1,
        curvature_safety_margin=0.0,
        contact_safety_margin=0.0,
    )
    projected = project_gravity_constraints(
        points,
        lengths,
        floor_height=-10.0,
        tube_radius=0.125,
        constraint_configuration=constraints,
        projection_configuration=GravityProjectionConfiguration(),
    )
    assert projected.converged
    assert projected.residuals.maximum_edge_error < 1e-7
    assert projected.residuals.separation_violation_count == 0
    assert projected.residuals.maximum_angle <= constraints.maximum_angle
    assert projected.residuals.minimum_floor_slack >= 0.0


def test_global_gravity_step_lowers_a_green_chain_to_the_floor() -> None:
    points = _horizontal_chain()
    configuration = replace(
        QuasistaticGravityConfiguration(),
        maximum_iterations=1,
    )
    result = minimize_gravitational_energy(points, configuration=configuration)
    assert result.final_energy < result.initial_energy
    assert len(result.iterations) == 1
    assert result.iterations[0].accepted
    expected_support = (
        result.floor_height
        + result.tube_radius
        + configuration.projection.validation_safety_margin
    )
    assert abs(np.min(result.points[:, 2]) - expected_support) < 1e-6
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(result.points, axis=0), axis=1),
        np.linalg.norm(np.diff(points, axis=0), axis=1),
        atol=2e-6,
    )
    assert gravitational_energy(result.points) < gravitational_energy(points)


def test_accepted_energy_is_monotone_and_runs_are_deterministic() -> None:
    points = _horizontal_chain()
    configuration = replace(
        QuasistaticGravityConfiguration(),
        maximum_iterations=3,
    )
    first = minimize_gravitational_energy(points, configuration=configuration)
    second = minimize_gravitational_energy(points, configuration=configuration)
    accepted = [item for item in first.iterations if item.accepted]
    assert all(item.energy_after < item.energy_before for item in accepted)
    np.testing.assert_array_equal(first.points, second.points)
    assert [item.accepted for item in first.iterations] == [
        item.accepted for item in second.iterations
    ]
    np.testing.assert_array_equal(
        [item.energy_after for item in first.iterations],
        [item.energy_after for item in second.iterations],
    )


def test_projection_failure_recomputes_with_a_smaller_trust_radius(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _horizontal_chain()
    residuals = GravityConstraintResiduals(
        maximum_edge_error=0.0,
        maximum_angle=0.0,
        minimum_contact_slack=np.inf,
        separation_violation_count=0,
        minimum_floor_slack=0.0,
    )

    def fail_projection(*args: object, **kwargs: object) -> GravityProjectionResult:
        candidate = np.asarray(args[0], dtype=np.float64)
        return GravityProjectionResult(
            points=candidate,
            converged=False,
            iterations=0,
            residuals=residuals,
            minimum_margin_slack=-1.0,
            message="deliberate projection failure",
            history=(),
        )

    monkeypatch.setattr(
        "autorigami.gravity.quasistatic.project_gravity_constraints",
        fail_projection,
    )
    configuration = replace(
        QuasistaticGravityConfiguration(),
        initial_trust_radius=0.05,
        minimum_trust_radius=0.02,
        maximum_iterations=1,
    )
    result = minimize_gravitational_energy(points, configuration=configuration)
    assert [item.trust_radius for item in result.iterations] == [0.05, 0.025]
    assert all(not item.accepted for item in result.iterations)
    assert "trust region exhausted" in result.message


def test_invalid_input_is_rejected_explicitly() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [0.34, 0.0, 0.0], [0.34, 0.34, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="requires a green input"):
        minimize_gravitational_energy(points)


def test_workload_reparametrizes_and_standard_validates() -> None:
    result = pack_under_gravity(
        _horizontal_chain(),
        configuration=replace(
            QuasistaticGravityConfiguration(), maximum_iterations=1
        ),
    )
    assert result.successful
    assert result.validation.valid
    assert result.material_points is result.optimization.points
    lengths = np.linalg.norm(np.diff(result.points, axis=0), axis=1)
    np.testing.assert_allclose(lengths[:-1], 0.34, atol=2e-5)


def test_workload_reports_an_invalid_reparametrized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = PolylineValidationResult(
        maximum_angle=1.0,
        curvature_violation_count=1,
        separation_violation_count=2,
        minimum_violating_distance=2.5,
        ignored_adjacent_edges=1,
        violation_masks=None,
    )
    monkeypatch.setattr(
        "autorigami.gravity.workloads.validate_polyline",
        lambda *args, **kwargs: invalid,
    )
    result = pack_under_gravity(
        _horizontal_chain(),
        configuration=replace(
            QuasistaticGravityConfiguration(), maximum_iterations=1
        ),
    )
    assert not result.successful
    assert result.validation is invalid
    assert "reparametrized quasistatic result is invalid" in result.message
