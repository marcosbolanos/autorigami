import numpy as np

from autorigami.geometry.curvature import get_polyline_angles
from autorigami.optimization.energies import (
    curvature_violation_energy_gradient,
    curvature_violation_energy,
    sparse_separation_energy,
)
from autorigami.optimization.workloads import (
    fix_curvature_violations,
    optimize_separation_violations,
)


def test_optimize_separation_violations_reduces_sparse_separation_energy() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    initial_energy = sparse_separation_energy(polyline, min_distance=0.25)
    optimized = optimize_separation_violations(
        polyline,
        min_distance=0.25,
        learning_rate=0.1,
        steps=10,
        n_ignored_adjacent_edges=1,
    )
    final_energy = sparse_separation_energy(optimized, min_distance=0.25)

    assert final_energy < initial_energy
    assert np.allclose(optimized[:, :2], polyline[:, :2])


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
        target_angle=0.25,
        learning_rate=0.1,
        steps=1,
        max_vertex_step=1.0,
    )

    assert np.all(np.linalg.norm(optimized - polyline, axis=1) > 0.0)
