import numpy as np

from autorigami._native import evaluate_tangent_point_exact


def _energy(polyline: np.ndarray) -> float:
    return float(
        evaluate_tangent_point_exact(
            np.asarray(polyline, dtype=np.float32),
            target_distance=1.3,
            attraction_strength=0.7,
            local_exclusion_length=1.8,
        )["energy"]
    )


def test_native_tangent_point_differential_matches_finite_differences() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.0, 0.4, 0.2],
            [2.0, 2.0, 0.3],
            [1.0, 2.2, 0.1],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    evaluation = evaluate_tangent_point_exact(polyline, 1.3, 0.7, 1.8)
    expected = np.empty(polyline.shape, dtype=np.float64)
    epsilon = np.float32(1e-3)
    for vertex in range(len(polyline)):
        for coordinate in range(3):
            positive = polyline.copy()
            negative = polyline.copy()
            positive[vertex, coordinate] += epsilon
            negative[vertex, coordinate] -= epsilon
            expected[vertex, coordinate] = (_energy(positive) - _energy(negative)) / (
                2.0 * float(epsilon)
            )

    np.testing.assert_allclose(
        evaluation["differential"],
        expected,
        rtol=2e-3,
        atol=2e-3,
    )


def test_tangent_point_energy_is_rigid_motion_invariant() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.0, 0.4, 0.2],
            [2.0, 2.0, 0.3],
            [1.0, 2.2, 0.1],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transformed = polyline @ rotation.T + np.array([3.0, -2.0, 1.0])

    assert np.isclose(_energy(polyline), _energy(transformed), rtol=2e-6)
