import numpy as np

from autorigami.geometry.curvature import get_polyline_angles
from autorigami.optimization.energies import (
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
