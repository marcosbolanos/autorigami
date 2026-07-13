import numpy as np
import pytest

from autorigami.geometry.curvature import get_polyline_angles
from autorigami.geometry.separation import check_self_intersections
import autorigami.optimization.workloads as workloads
from autorigami.optimization.energies import (
    curvature_violation_energy_gradient,
    curvature_violation_energy,
)
from autorigami.optimization.workloads import (
    fix_curvature_violations,
    fix_separation_violations,
)


def test_fix_separation_violations_corrects_and_reparametrizes_a_hairpin() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.1, 2.0, 0.0],
            [0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )

    initial_validation = check_self_intersections(
        polyline,
        min_euclid_distance=0.25,
        n_ignored_adjacent_edges=2,
    )
    optimized = fix_separation_violations(
        polyline,
        min_distance=0.25,
        edge_length=0.25,
        coordinates=["x", "y"],
        n_ignored_adjacent_edges=2,
    )
    final_validation = check_self_intersections(
        optimized,
        min_euclid_distance=0.25,
        n_ignored_adjacent_edges=2,
    )

    assert initial_validation["edges"] is not None
    assert final_validation["edges"] is None
    assert len(optimized) > len(polyline)
    edge_lengths = np.linalg.norm(np.diff(optimized, axis=0), axis=1)
    assert np.all(edge_lengths <= 0.25 + 1e-5)
    assert np.isclose(np.median(edge_lengths), 0.25, atol=1e-5)
    assert edge_lengths[-1] <= 0.25 + 1e-5


def test_fix_separation_violations_retries_failed_validation(monkeypatch) -> None:
    polyline = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    validation_count = 0

    def fail_once(*args, **kwargs):
        nonlocal validation_count
        validation_count += 1
        if validation_count == 1:
            return {"edges": [(0, 1)], "distances": [0.0]}
        return {"edges": None, "distances": None}

    monkeypatch.setattr(workloads, "check_self_intersections", fail_once)

    fix_separation_violations(
        polyline,
        min_distance=0.25,
        edge_length=0.25,
        validation_attempts=2,
    )

    assert validation_count == 2


def test_fix_separation_violations_warns_when_validation_never_succeeds(
    monkeypatch,
) -> None:
    polyline = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    def always_fail(*args, **kwargs):
        return {"edges": [(0, 1)], "distances": [0.0]}

    monkeypatch.setattr(workloads, "check_self_intersections", always_fail)

    with pytest.warns(RuntimeWarning, match="1 separation violations remain"):
        fix_separation_violations(
            polyline,
            min_distance=0.25,
            edge_length=0.25,
            validation_attempts=2,
        )


def test_fix_curvature_violations_reduces_curvature_energy() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    target_angle = 0.25

    initial_energy = curvature_violation_energy(polyline, target_angle)
    initial_max_angle = float(np.max(get_polyline_angles(polyline)))
    optimized = fix_curvature_violations(
        polyline,
        target_angle=target_angle,
        learning_rate=0.5,
        steps=10,
        max_vertex_step=0.1,
        edge_length=1.0,
    )
    final_energy = curvature_violation_energy(optimized, target_angle)
    final_max_angle = float(np.max(get_polyline_angles(optimized)))

    assert final_energy < initial_energy
    assert final_max_angle < initial_max_angle


def test_curvature_energy_gradient_matches_finite_differences() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [1.4, 1.1, -0.2],
            [2.2, 1.5, 0.3],
        ],
        dtype=np.float32,
    )
    target_angle = 0.2
    gradient = curvature_violation_energy_gradient(polyline, target_angle)
    finite_difference_gradient = np.zeros_like(polyline)
    epsilon = np.float32(1e-3)

    for vertex_index in range(len(polyline)):
        for coordinate_index in range(3):
            positive = polyline.copy()
            negative = polyline.copy()
            positive[vertex_index, coordinate_index] += epsilon
            negative[vertex_index, coordinate_index] -= epsilon
            finite_difference_gradient[vertex_index, coordinate_index] = (
                curvature_violation_energy(positive, target_angle)
                - curvature_violation_energy(negative, target_angle)
            ) / (2.0 * epsilon)

    assert np.allclose(gradient, finite_difference_gradient, atol=2e-4, rtol=2e-3)


def test_curvature_optimization_moves_all_participating_vertices() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    optimized = fix_curvature_violations(
        polyline,
        target_angle=0.5,
        edge_length=1.0,
        learning_rate=0.1,
        steps=10,
        max_vertex_step=1.0,
    )

    assert np.all(np.linalg.norm(optimized - polyline, axis=1) > 0.0)


def test_fix_curvature_violations_warns_when_validation_fails() -> None:
    polyline = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.warns(RuntimeWarning, match="1 curvature violations remain"):
        fix_curvature_violations(
            polyline,
            target_angle=0.25,
            edge_length=1.0,
            steps=0,
            validation_attempts=2,
        )
