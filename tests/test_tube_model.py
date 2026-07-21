import numpy as np

from autorigami.spiral_generation.tube_model import (
    TUBE_LENGTH_NM,
    TUBE_RADIUS_NM,
    generate_y_centerlines,
    tube_model_mesh,
)


def test_y_centerlines_have_equal_lengths_and_angles() -> None:
    centerlines = generate_y_centerlines()
    directions = np.stack([line[1] - line[0] for line in centerlines])

    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), TUBE_LENGTH_NM)
    normalized = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    pairwise_cosines = normalized @ normalized.T
    np.testing.assert_allclose(
        pairwise_cosines[np.triu_indices(3, k=1)],
        np.cos(np.deg2rad(120.0)),
        atol=1e-6,
    )


def test_y_tubes_use_the_dna_radius() -> None:
    mesh = tube_model_mesh()

    assert len(mesh) == 3
    bottom_tube = mesh[0]
    assert bottom_tube is not None
    extents = np.ptp(bottom_tube.points, axis=0)
    assert np.isclose(extents[0], 2.0 * TUBE_RADIUS_NM, atol=1e-3)
    assert np.isclose(extents[1], 2.0 * TUBE_RADIUS_NM, atol=1e-3)
    assert np.isclose(extents[2], TUBE_LENGTH_NM, atol=1e-3)
