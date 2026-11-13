import numpy as np
import pytest
from scipy.spatial import cKDTree

from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.geometry.validation import validate_polyline
from autorigami.packing import pack_under_gravity
from autorigami.types import Polyline


def _spiral(
    start_radius: float, end_radius: float, pitch: float, turns: float
) -> Polyline:
    """A conical spiral along z, sampled at the base-pair interval."""
    position = np.linspace(0.0, 1.0, 20_000)
    angle = 2.0 * np.pi * turns * position
    radius = start_radius + (end_radius - start_radius) * position
    curve = np.column_stack(
        (radius * np.cos(angle), radius * np.sin(angle), pitch * turns * position)
    ).astype(np.float32)
    return reparametrize_arc_length(curve, 0.34)


@pytest.mark.parametrize(
    ("start_radius", "end_radius", "pitch"),
    [
        (16.0, 16.0, 5.2),  # loose tube
        (10.0, 22.0, 4.0),  # loose funnel
        (8.0, 8.0, 2.6),  # narrow tube that touches the separation limit
    ],
)
def test_packing_gives_a_valid_tighter_curve_with_the_same_horizontal_shape(
    start_radius: float, end_radius: float, pitch: float
) -> None:
    spiral = _spiral(start_radius, end_radius, pitch, turns=6.0)

    result = pack_under_gravity(spiral)

    assert validate_polyline(result.points).valid
    assert result.validation.valid
    horizontal_distance, _ = cKDTree(spiral[:, :2]).query(result.points[:, :2])
    assert float(horizontal_distance.max()) < 0.2
    if pitch > 3.0:
        assert result.final_height < 0.75 * result.initial_height
    if start_radius == end_radius:
        # Five turns stack on the anchored first turn at the contact distance.
        assert result.final_height == pytest.approx(5.0 * 2.65 + pitch, abs=1.5)


def test_packing_rejects_a_curve_with_less_than_two_turns() -> None:
    with pytest.raises(AssertionError, match="more than one turn"):
        pack_under_gravity(_spiral(16.0, 16.0, 5.2, turns=0.5))
