import numpy as np

from autorigami.spiral_generation.full_double_spiral import (
    SpiralBase,
    SpiralMiddleFunnel,
    generate_full_spiral,
)


def test_generate_full_spiral_concatenates_middle_funnel() -> None:
    polyline = generate_full_spiral()

    assert polyline.shape == (29998, 3)
    assert polyline[9999, 2] > 0.0
    assert polyline[19998, 2] > polyline[9999, 2]
    assert polyline[-1, 2] > polyline[19998, 2]


def test_discretize_returns_absolute_final_angle_for_chaining() -> None:
    base = SpiralBase(starting_angle=1.0, radius=2.0, winding_frequency=3.0, length=4.0)
    _, base_final_angle = base.discretize(10)

    middle = SpiralMiddleFunnel(
        starting_angle=base_final_angle,
        starting_coords=np.array([0.0, 0.0, 4.0], dtype=np.float32),
        starting_x_radius=2.0,
        x_radius_increase_factor=2.0,
        y_radius=2.0,
        winding_frequency=3.0,
        length=4.0,
    )
    _, middle_final_angle = middle.discretize(10)

    assert np.isclose(base_final_angle, 1.0 + 6.0 * np.pi)
    assert np.isclose(middle_final_angle, 1.0 + 12.0 * np.pi)
