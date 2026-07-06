from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from autorigami.parametrization import PiecewiseHermite, Polyline
from autorigami.types import FloatArray


@dataclass(frozen=True, slots=True)
class ControlPointSpiralDesign:
    """DNA path controlled by a vertically ordered centerline."""

    control_points: FloatArray
    radius: float
    turns: float
    sample_count: int

    def __post_init__(self) -> None:
        if self.control_points.ndim != 2 or self.control_points.shape[1] != 3:
            raise ValueError("control_points must have shape (N, 3)")
        if self.control_points.shape[0] < 2:
            raise ValueError("control_points must contain at least two points")
        if not np.all(np.isfinite(self.control_points)):
            raise ValueError("control_points must be finite")
        if not np.all(np.diff(self.control_points[:, 2]) > 0.0):
            raise ValueError("control point z values must be strictly increasing")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be a finite positive number")
        if not np.isfinite(self.turns) or self.turns <= 0.0:
            raise ValueError("turns must be a finite positive number")
        if self.sample_count < 2:
            raise ValueError("sample_count must be >= 2")


@dataclass(frozen=True, slots=True)
class SpiralValidationSummary:
    curvature_passes: bool
    approximate_max_curvature: float
    approximate_min_curvature_radius: float
    approximate_min_separation: float
    separation_passes: bool


def default_control_points() -> FloatArray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, -1.0, 8.0],
            [-1.5, 1.5, 16.0],
            [0.0, 0.0, 24.0],
        ],
        dtype=np.float64,
    )


def generate_control_point_spiral(design: ControlPointSpiralDesign) -> Polyline:
    parameters = _normalized_z_parameters(design.control_points[:, 2])
    sample_parameters = np.linspace(0.0, 1.0, design.sample_count, dtype=np.float64)

    centerline = np.column_stack(
        [
            _spline_sample(parameters, design.control_points[:, 0], sample_parameters),
            _spline_sample(parameters, design.control_points[:, 1], sample_parameters),
            _spline_sample(parameters, design.control_points[:, 2], sample_parameters),
        ]
    )
    theta = sample_parameters * design.turns * 2.0 * np.pi
    offsets = np.column_stack(
        [
            design.radius * np.cos(theta),
            design.radius * np.sin(theta),
            np.zeros_like(theta),
        ]
    )
    return Polyline(points=np.asarray(centerline + offsets, dtype=np.float64))


def generate_natural_control_curve(
    control_points: FloatArray,
    sample_count: int,
) -> Polyline:
    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError("control_points must have shape (N, 3)")
    if control_points.shape[0] < 2:
        raise ValueError("control_points must contain at least two points")
    if not np.all(np.isfinite(control_points)):
        raise ValueError("control_points must be finite")
    if not np.all(np.diff(control_points[:, 2]) > 0.0):
        raise ValueError("control point z values must be strictly increasing")
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2")

    z_values = np.asarray(control_points[:, 2], dtype=np.float64)
    sample_z = np.linspace(z_values[0], z_values[-1], sample_count, dtype=np.float64)
    if control_points.shape[0] == 2:
        x_values = np.interp(sample_z, z_values, control_points[:, 0])
        y_values = np.interp(sample_z, z_values, control_points[:, 1])
    else:
        x_values = CubicSpline(z_values, control_points[:, 0], bc_type="natural")(
            sample_z
        )
        y_values = CubicSpline(z_values, control_points[:, 1], bc_type="natural")(
            sample_z
        )
    return Polyline(
        points=np.column_stack([x_values, y_values, sample_z]).astype(np.float64)
    )


def piecewise_hermite_from_polyline(polyline: Polyline) -> PiecewiseHermite:
    points = np.asarray(polyline.points, dtype=np.float64)
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        tangents[1:-1] = 0.5 * (points[2:] - points[:-2])
    return PiecewiseHermite(points=points, tangents=tangents)


def summarize_spiral_validation(
    polyline: Polyline,
    curvature_passes: bool,
    min_separation: float,
    neighbor_skip: int,
) -> SpiralValidationSummary:
    if min_separation <= 0.0:
        raise ValueError("min_separation must be > 0")
    if neighbor_skip < 1:
        raise ValueError("neighbor_skip must be >= 1")

    approximate_min_separation = approximate_self_separation(
        polyline=polyline,
        neighbor_skip=neighbor_skip,
    )
    return SpiralValidationSummary(
        curvature_passes=curvature_passes,
        approximate_max_curvature=approximate_polyline_max_curvature(polyline),
        approximate_min_curvature_radius=approximate_polyline_min_curvature_radius(
            polyline
        ),
        approximate_min_separation=approximate_min_separation,
        separation_passes=approximate_min_separation >= min_separation,
    )


def approximate_polyline_max_curvature(polyline: Polyline) -> float:
    points = polyline.points
    if points.shape[0] < 3:
        return 0.0

    before = points[:-2]
    middle = points[1:-1]
    after = points[2:]
    side_a = np.linalg.norm(middle - before, axis=1)
    side_b = np.linalg.norm(after - middle, axis=1)
    side_c = np.linalg.norm(after - before, axis=1)
    cross = np.linalg.norm(np.cross(middle - before, after - before), axis=1)
    denominator = side_a * side_b * side_c
    valid = denominator > 0.0
    if not np.any(valid):
        return 0.0
    curvatures = np.zeros_like(denominator)
    curvatures[valid] = 2.0 * cross[valid] / denominator[valid]
    return float(np.max(curvatures))


def approximate_polyline_min_curvature_radius(polyline: Polyline) -> float:
    max_curvature = approximate_polyline_max_curvature(polyline)
    if max_curvature == 0.0:
        return np.inf
    return 1.0 / max_curvature


def approximate_self_separation(polyline: Polyline, neighbor_skip: int) -> float:
    if neighbor_skip < 1:
        raise ValueError("neighbor_skip must be >= 1")

    points = polyline.points
    best = np.inf
    for index in range(points.shape[0]):
        later = points[index + neighbor_skip + 1 :]
        if later.shape[0] == 0:
            continue
        distances = np.linalg.norm(later - points[index], axis=1)
        best = min(best, float(np.min(distances)))
    return best


def _normalized_z_parameters(z_values: FloatArray) -> FloatArray:
    z_span = z_values[-1] - z_values[0]
    if z_span <= 0.0:
        raise ValueError("control point z values must span a positive range")
    return np.asarray((z_values - z_values[0]) / z_span, dtype=np.float64)


def _spline_sample(
    parameters: FloatArray,
    values: FloatArray,
    sample_parameters: FloatArray,
) -> FloatArray:
    if parameters.shape[0] == 2:
        return np.interp(sample_parameters, parameters, values).astype(np.float64)
    return np.asarray(
        CubicSpline(parameters, values, bc_type="natural")(sample_parameters),
        dtype=np.float64,
    )
