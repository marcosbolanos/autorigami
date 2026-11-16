from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import KDTree

from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.spiral_generation import (
    FunnelSegment,
    Spiral,
    StraightSegment,
    YSegment,
    load_curves,
    validate_curves,
)
from autorigami.spiral_generation.validation import cross_curve_separation
from autorigami.spiral_generation.three_way_tube_v1 import generate_full_spiral
from autorigami.viz import show_curves, show_spiral


def _largest_edge(curve: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).max())


def test_straight_segment_is_an_elliptic_helix() -> None:
    spiral = Spiral()
    segment = spiral.add_straight_segment(26.0, x_radius=16.0, y_radius=10.0)

    assert isinstance(segment, StraightSegment)
    assert spiral.segments == [segment]
    assert len(segment.termini) == 1
    (curve,) = spiral.curves()
    assert curve.dtype == np.float32
    assert np.allclose(
        (curve[:, 0] / 16.0) ** 2 + (curve[:, 1] / 10.0) ** 2, 1.0, atol=1e-3
    )
    assert curve[0, 2] == 0.0 and curve[-1, 2] == pytest.approx(26.0)
    # 10 turns: the curve returns to its first horizontal position.
    assert np.allclose(curve[-1, :2], curve[0, :2], atol=1e-3)
    assert segment.termini[0].height == pytest.approx(26.0)


def test_funnel_segment_interpolates_the_radii() -> None:
    spiral = Spiral()
    stem = spiral.add_straight_segment(13.0, x_radius=16.0, y_radius=16.0)
    funnel = stem.termini[0].add_funnel_segment(
        26.0, end_x_radius=26.0, end_y_radius=16.0
    )

    assert isinstance(funnel, FunnelSegment)
    assert spiral.segments == [stem, funnel]
    assert stem.termini[0].next_segment is funnel
    (curve,) = spiral.curves()
    assert _largest_edge(curve) <= 0.345
    middle = curve[np.abs(curve[:, 2] - 26.0) < 1.3]
    assert np.abs(middle[:, 0]).max() == pytest.approx(21.0, abs=0.6)
    assert np.abs(middle[:, 1]).max() == pytest.approx(16.0, abs=0.2)
    assert funnel.termini[0].x_radius == 26.0


def test_spiral_can_start_with_a_funnel() -> None:
    spiral = Spiral()
    spiral.add_funnel_segment(
        13.0,
        start_x_radius=10.0,
        start_y_radius=10.0,
        end_x_radius=20.0,
        end_y_radius=20.0,
    )

    (curve,) = spiral.curves()
    radius = np.hypot(curve[:, 0], curve[:, 1])
    assert radius[0] == pytest.approx(10.0) and radius[-1] == pytest.approx(
        20.0, abs=1e-3
    )


def test_y_segment_follows_design_1_and_has_two_termini() -> None:
    spiral = Spiral()
    stem = spiral.add_straight_segment(40.0, x_radius=16.0, y_radius=16.0)
    y = stem.termini[0].add_y_segment(
        middle_length=40.0,
        top_length=40.0,
        middle_opening_angle_deg=38.66,
        top_opening_angle_deg=23.71,
    )

    assert isinstance(y, YSegment)
    assert len(y.termini) == 2
    assert not y.termini[0].starts_new_curve and y.termini[1].starts_new_curve
    (curve,) = spiral.curves()
    assert _largest_edge(curve) <= 0.345
    design_1 = reparametrize_arc_length(generate_full_spiral(), 0.34)
    lower = curve[curve[:, 2] < 80.0]
    assert KDTree(design_1).query(lower)[0].max() < 0.2
    # The curve ends at the innermost point of the first arm.
    first_arm = y.termini[0]
    assert np.linalg.norm(curve[-1, :2] - first_arm.center) == pytest.approx(
        16.0, abs=0.05
    )
    assert np.linalg.norm(first_arm.center) > np.linalg.norm(curve[-1, :2])


def test_segments_on_the_two_arms_give_two_curves() -> None:
    spiral = Spiral()
    stem = spiral.add_straight_segment(13.0, x_radius=16.0, y_radius=16.0)
    y = stem.termini[0].add_y_segment(
        middle_length=26.0,
        top_length=26.0,
        middle_opening_angle_deg=38.66,
        top_opening_angle_deg=23.71,
    )
    first = y.termini[0].add_straight_segment(13.0)
    second = y.termini[1].add_straight_segment(13.0)

    assert spiral.segments == [stem, y, first, second]
    main, branch = spiral.curves()
    assert _largest_edge(main) <= 0.345
    assert _largest_edge(branch) <= 0.345
    assert main[-1, 2] == pytest.approx(13.0 + 26.0 + 26.0 + 13.0, abs=1e-3)
    # The arms are on opposite sides of the stem axis and lean outward.
    assert float(first.termini[0].center @ second.termini[0].center) < 0.0
    assert np.linalg.norm(first.termini[0].center) > np.linalg.norm(y.termini[0].center)
    # The second curve starts above the last turn of its arm.
    distance = KDTree(main).query(branch)[0]
    assert distance.min() > 2.0


def test_a_terminus_accepts_one_segment_only() -> None:
    spiral = Spiral()
    stem = spiral.add_straight_segment(13.0, x_radius=16.0, y_radius=16.0)
    stem.termini[0].add_straight_segment(13.0)

    with pytest.raises(AssertionError):
        stem.termini[0].add_straight_segment(13.0)
    with pytest.raises(AssertionError):
        spiral.add_straight_segment(13.0, x_radius=16.0, y_radius=16.0)


def test_validate_examines_each_curve_and_each_pair_of_curves() -> None:
    spiral = Spiral()
    stem = spiral.add_straight_segment(
        13.0, x_radius=16.0, y_radius=16.0, rise_per_turn=2.65
    )
    assert spiral.validate().valid

    y = stem.termini[0].add_y_segment(
        middle_length=26.0,
        top_length=26.0,
        middle_opening_angle_deg=38.66,
        top_opening_angle_deg=23.71,
    )
    y.termini[0].add_straight_segment(13.0)
    y.termini[1].add_straight_segment(13.0)
    result = spiral.validate()
    assert len(result.curves) == 2
    assert [(pair.first_curve, pair.second_curve) for pair in result.cross_curve] == [
        (0, 1)
    ]
    assert not result.valid


def test_cross_curve_separation_counts_only_pairs_across_the_curves() -> None:
    x = np.arange(0.0, 30.0, 0.34, dtype=np.float32)
    first = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    offset = np.array([0.0, 2.0, 0.0], dtype=np.float32)

    near = cross_curve_separation(first, first + offset)
    far = cross_curve_separation(first, first + 1.5 * offset)

    assert near.violation_count > 0
    assert near.minimum_violating_distance == pytest.approx(2.0)
    assert far.violation_count == 0 and far.minimum_violating_distance is None
    assert validate_curves([first, first + 1.5 * offset]).valid
    assert not validate_curves([first, first + offset]).valid


def test_save_and_load_curves(tmp_path: Path) -> None:
    spiral = Spiral()
    stem = spiral.add_straight_segment(13.0, x_radius=16.0, y_radius=16.0)
    y = stem.termini[0].add_y_segment(
        middle_length=26.0,
        top_length=26.0,
        middle_opening_angle_deg=38.66,
        top_opening_angle_deg=23.71,
    )
    y.termini[1].add_straight_segment(13.0)

    spiral.save(tmp_path / "spiral.npz")
    loaded = load_curves(tmp_path / "spiral.npz")

    assert len(loaded) == 2
    for saved, curve in zip(loaded, spiral.curves(), strict=True):
        assert saved.dtype == np.float32 and np.array_equal(saved, curve)


def test_show_spiral_has_one_line_for_each_segment() -> None:
    spiral = Spiral()
    stem = spiral.add_straight_segment(13.0, x_radius=16.0, y_radius=16.0)
    funnel = stem.termini[0].add_funnel_segment(
        13.0, end_x_radius=20.0, end_y_radius=20.0
    )

    figure = show_spiral(spiral, highlight=funnel)

    assert [trace["name"] for trace in figure.to_dict()["data"]] == [
        "0: StraightSegment",
        "1: FunnelSegment",
    ]
    assert len(show_curves(spiral.curves()).to_dict()["data"]) == 1
