import numpy as np

from autorigami.spiral_generation.full_double_spiral import (
    SpiralBase,
    MiddleSegment,
    generate_full_spiral,
)


def test_generate_full_spiral_concatenates_segments() -> None:
    polyline = generate_full_spiral()

    assert polyline.shape == (29998, 3)
    assert polyline[9999, 2] > 0.0
    assert polyline[19998, 2] > polyline[9999, 2]
    assert polyline[-1, 2] > polyline[19998, 2]


def test_get_points_accepts_scalar_and_array_inputs() -> None:
    base = SpiralBase(orientation_angle=1.0, radius=2.0, turns=3.0, height=4.0)
    positions = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    base_polyline = base.get_points(positions)
    base_point = base.get_points(np.float32(0.5))

    assert base_polyline.shape == (10, 3)
    assert base_point.shape == (3,)

    base_final_angle = float(base.orientation_angle + 2.0 * np.pi * base.turns)

    middle = MiddleSegment(
        orientation_angle=base_final_angle,
        anchor_point=np.array([0.0, 0.0, 4.0], dtype=np.float32),
        start_x_radius=2.0,
        end_x_radius_scale=2.0,
        y_radius=2.0,
        turns=3.0,
        height=4.0,
    )
    middle_polyline = middle.get_points(positions)
    middle_point = middle.get_points(0.5)
    middle_final_angle = float(middle.orientation_angle + 2.0 * np.pi * middle.turns)

    assert middle_polyline.shape == (10, 3)
    assert middle_point.shape == (3,)
    assert np.isclose(base_final_angle, 1.0 + 6.0 * np.pi)
    assert np.isclose(middle_final_angle, 1.0 + 12.0 * np.pi)
