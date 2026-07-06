from __future__ import annotations

import numpy as np
import pytest

from autorigami._native import validate_piecewise_curve_curvature
from autorigami.spiral_generation.control_point_spiral import (
    ControlPointSpiralDesign,
    approximate_polyline_min_curvature_radius,
    approximate_polyline_max_curvature,
    approximate_self_separation,
    default_control_points,
    generate_control_point_spiral,
    generate_natural_control_curve,
    piecewise_hermite_from_polyline,
    summarize_spiral_validation,
)


def test_generate_control_point_spiral_returns_requested_samples() -> None:
    design = ControlPointSpiralDesign(
        control_points=default_control_points(),
        radius=1.5,
        turns=4.0,
        sample_count=128,
    )

    polyline = generate_control_point_spiral(design)

    assert polyline.points.shape == (128, 3)
    assert np.all(np.isfinite(polyline.points))


def test_control_points_must_be_strictly_vertical_ordered() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="z values must be strictly increasing"):
        ControlPointSpiralDesign(
            control_points=points,
            radius=1.0,
            turns=2.0,
            sample_count=32,
        )


def test_generated_spiral_can_use_native_curvature_validation() -> None:
    design = ControlPointSpiralDesign(
        control_points=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 12.0],
            ],
            dtype=np.float64,
        ),
        radius=1.0,
        turns=1.0,
        sample_count=64,
    )

    polyline = generate_control_point_spiral(design)
    hermite = piecewise_hermite_from_polyline(polyline)

    assert validate_piecewise_curve_curvature(
        piecewise_hermite=hermite,
        max_curvature=100.0,
        curvature_tolerance=1e-6,
    )


def test_generate_natural_control_curve_interpolates_endpoints() -> None:
    control_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 1.0, 8.0],
            [-2.0, 3.0, 16.0],
        ],
        dtype=np.float64,
    )

    polyline = generate_natural_control_curve(control_points, sample_count=80)

    assert polyline.points.shape == (80, 3)
    assert np.allclose(polyline.points[0], control_points[0])
    assert np.allclose(polyline.points[-1], control_points[-1])
    assert np.all(np.diff(polyline.points[:, 2]) > 0.0)


def test_straight_natural_control_curve_has_infinite_curvature_radius() -> None:
    polyline = generate_natural_control_curve(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 12.0],
            ],
            dtype=np.float64,
        ),
        sample_count=32,
    )

    assert approximate_polyline_max_curvature(polyline) == 0.0
    assert np.isinf(approximate_polyline_min_curvature_radius(polyline))


def test_validation_summary_reports_approximate_metrics() -> None:
    design = ControlPointSpiralDesign(
        control_points=default_control_points(),
        radius=1.2,
        turns=3.0,
        sample_count=96,
    )
    polyline = generate_control_point_spiral(design)

    summary = summarize_spiral_validation(
        polyline=polyline,
        curvature_passes=True,
        min_separation=0.1,
        neighbor_skip=8,
    )

    assert summary.curvature_passes
    assert summary.approximate_max_curvature == approximate_polyline_max_curvature(
        polyline
    )
    assert (
        summary.approximate_min_curvature_radius
        == approximate_polyline_min_curvature_radius(polyline)
    )
    assert summary.approximate_min_separation == approximate_self_separation(
        polyline,
        neighbor_skip=8,
    )
    assert summary.separation_passes
