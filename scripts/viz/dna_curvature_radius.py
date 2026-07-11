import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from autorigami.mesh_io import (
    CYTOSINE_COLOR,
    dna_molecule_line_segments_from_base_pair_centers,
    dna_molecule_mesh_from_base_pair_centers,
)
from autorigami.types import Polyline

BASE_COUNT = 60
FIRST_TURNING_BASE_COUNT = 40
TURN_ANGLE = 0.057
DISTANCE_BETWEEN_BASE_PAIRS = 0.34
DNA_MOLECULE_RADIUS = 1.05
DNA_SEGMENT_RADIUS = 0.05
CURVATURE_MARKER_COLOR = tuple(int(channel) for channel in CYTOSINE_COLOR[:3])
EQUATION_LABEL_HEIGHT = 1.05
EQUATION_LABEL_LINE_COUNT = 2
EQUATION_LABEL_GAP = 0.35


def generate_s_shaped_base_pair_centers(
    base_count: int = BASE_COUNT,
    first_turning_base_count: int = FIRST_TURNING_BASE_COUNT,
    turn_angle: float = TURN_ANGLE,
    distance_between_base_pairs: float = DISTANCE_BETWEEN_BASE_PAIRS,
) -> Polyline:
    assert base_count >= 2, "base_count must be at least 2"
    assert 0 < first_turning_base_count < base_count, (
        "first_turning_base_count must split the base sequence"
    )
    assert turn_angle > 0.0, "turn_angle must be positive"
    assert distance_between_base_pairs > 0.0, (
        "distance_between_base_pairs must be positive"
    )

    turn_signs = np.ones(base_count, dtype=np.float32)
    turn_signs[first_turning_base_count:] = -1.0
    headings = np.cumsum(turn_signs * np.float32(turn_angle), dtype=np.float32)
    interval_headings = 0.5 * (headings[:-1] + headings[1:])
    intervals = distance_between_base_pairs * np.column_stack(
        (
            np.cos(interval_headings),
            np.sin(interval_headings),
            np.zeros(base_count - 1, dtype=np.float32),
        )
    )
    centers = np.vstack(
        (np.zeros((1, 3), dtype=np.float32), np.cumsum(intervals, axis=0))
    )
    centers -= 0.5 * (centers[0] + centers[-1])
    return centers.astype(np.float32)


def generate_curvature_circle(
    base_pair_centers: Polyline,
    first_turning_base_count: int = FIRST_TURNING_BASE_COUNT,
    turn_angle: float = TURN_ANGLE,
    distance_between_base_pairs: float = DISTANCE_BETWEEN_BASE_PAIRS,
    point_count: int = 240,
) -> tuple[Polyline, Polyline, float]:
    assert base_pair_centers.ndim == 2 and base_pair_centers.shape[1] == 3, (
        "base_pair_centers must have shape (n, 3)"
    )
    assert len(base_pair_centers) > first_turning_base_count, (
        "base_pair_centers must contain the first turning arc"
    )
    assert first_turning_base_count >= 2, (
        "first_turning_base_count must be at least 2"
    )
    assert turn_angle > 0.0, "turn_angle must be positive"
    assert distance_between_base_pairs > 0.0, (
        "distance_between_base_pairs must be positive"
    )
    assert point_count >= 16, "point_count must be at least 16"

    curvature_radius = distance_between_base_pairs / turn_angle
    first_arc_centers = base_pair_centers[:first_turning_base_count]
    first_segment = first_arc_centers[1] - first_arc_centers[0]
    tangent = first_segment / np.linalg.norm(first_segment)
    left_normal = np.array([-tangent[1], tangent[0], 0.0], dtype=np.float32)
    circle_center = first_arc_centers[0] + curvature_radius * left_normal
    first_radius = first_arc_centers[0] - circle_center
    first_radius_angle = math.atan2(float(first_radius[1]), float(first_radius[0]))
    circle_angles = np.linspace(
        0.0,
        2.0 * math.pi,
        point_count,
        endpoint=False,
        dtype=np.float32,
    )
    circle = circle_center + curvature_radius * np.column_stack(
        (
            np.cos(circle_angles),
            np.sin(circle_angles),
            np.zeros(point_count, dtype=np.float32),
        )
    )
    arc_angle_span = turn_angle * float(first_turning_base_count - 1)
    radius_angle = first_radius_angle - 0.25 * arc_angle_span
    radius_endpoint = circle_center + curvature_radius * np.array(
        [math.cos(radius_angle), math.sin(radius_angle), 0.0], dtype=np.float32
    )
    radius_segment = np.vstack((circle_center, radius_endpoint)).astype(np.float32)
    return circle.astype(np.float32), radius_segment, curvature_radius


def pyvista_lines_from_segments(
    starts: Polyline,
    ends: Polyline,
    colors: npt.NDArray[np.uint8],
) -> pv.PolyData:
    assert starts.ndim == 2 and starts.shape[1] == 3, "starts must have shape (n, 3)"
    assert ends.shape == starts.shape, "ends must have the same shape as starts"
    assert colors.shape == (len(starts), 4), "colors must have shape (n, 4)"

    points = np.empty((2 * len(starts), 3), dtype=np.float32)
    points[0::2] = starts
    points[1::2] = ends
    line_indices = np.arange(2 * len(starts), dtype=np.int64).reshape((-1, 2))
    lines = np.column_stack((np.full(len(starts), 2, dtype=np.int64), line_indices))
    polydata = pv.PolyData(points)
    polydata.lines = lines.ravel()
    polydata.point_data["RGBA"] = np.repeat(colors, 2, axis=0)
    return polydata


def add_radius_marker(
    plotter: pv.Plotter,
    radius_segment: Polyline,
    mark_length: float,
) -> None:
    assert radius_segment.shape == (2, 3), "radius_segment must have shape (2, 3)"
    assert mark_length > 0.0, "mark_length must be positive"

    direction = radius_segment[1] - radius_segment[0]
    direction = direction / np.linalg.norm(direction)
    mark_direction = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
    for endpoint in radius_segment:
        mark = np.vstack(
            (
                endpoint - 0.5 * mark_length * mark_direction,
                endpoint + 0.5 * mark_length * mark_direction,
            )
        )
        plotter.add_mesh(  # type: ignore
            pv.lines_from_points(mark),
            color=CURVATURE_MARKER_COLOR,
            line_width=7,
            render_lines_as_tubes=True,
        )


def render_equation_texture() -> npt.NDArray[np.uint8]:
    figure = Figure(figsize=(6.0, 3.6), dpi=200)
    figure.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(figure)
    color = tuple(float(channel) / 255.0 for channel in CURVATURE_MARKER_COLOR)
    figure.text(
        0.5,
        0.68,
        r"$R = \frac{1}{\kappa_{min}}$",
        color=color,
        fontsize=76,
        ha="center",
        va="center",
    )
    figure.text(
        0.5,
        0.28,
        r"$\approx 6\,\mathrm{nm}$",
        color=color,
        fontsize=76,
        ha="center",
        va="center",
    )
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    alpha = image[:, :, 3]
    occupied = np.argwhere(alpha > 0)
    assert occupied.size > 0, "equation texture must contain rendered text"
    y_min, x_min = occupied.min(axis=0)
    y_max, x_max = occupied.max(axis=0)
    padding = 10
    y_min = max(int(y_min) - padding, 0)
    x_min = max(int(x_min) - padding, 0)
    y_max = min(int(y_max) + padding, image.shape[0] - 1)
    x_max = min(int(x_max) + padding, image.shape[1] - 1)
    return image[y_min : y_max + 1, x_min : x_max + 1].copy()


def equation_label_mesh(
    radius_segment: Polyline,
    dna_reference_points: Polyline,
    label_texture: npt.NDArray[np.uint8],
    label_height: float,
    label_gap: float,
) -> pv.PolyData:
    assert radius_segment.shape == (2, 3), "radius_segment must have shape (2, 3)"
    assert dna_reference_points.ndim == 2 and dna_reference_points.shape[1] == 3, (
        "dna_reference_points must have shape (n, 3)"
    )
    assert label_texture.ndim == 3 and label_texture.shape[2] == 4, (
        "label_texture must have shape (height, width, 4)"
    )
    assert label_height > 0.0, "label_height must be positive"
    assert label_gap > 0.0, "label_gap must be positive"

    direction = radius_segment[1] - radius_segment[0]
    direction = direction / np.linalg.norm(direction)
    perpendicular_direction = np.array(
        [-direction[1], direction[0], 0.0], dtype=np.float32
    )
    aspect_ratio = float(label_texture.shape[1]) / float(label_texture.shape[0])
    label_width = label_height * aspect_ratio

    candidate_height_directions = (perpendicular_direction, -perpendicular_direction)
    candidate_centers = [
        radius_segment[0]
        + 0.5 * label_width * direction
        + (label_gap + 0.5 * label_height) * candidate_height_direction
        for candidate_height_direction in candidate_height_directions
    ]
    minimum_distances = [
        float(np.min(np.linalg.norm(dna_reference_points - center, axis=1)))
        for center in candidate_centers
    ]
    height_direction = candidate_height_directions[int(np.argmax(minimum_distances))]
    center = (
        radius_segment[0]
        + 0.5 * label_width * direction
        + (label_gap + 0.5 * label_height) * height_direction
    )
    half_width = 0.5 * label_width * direction
    half_height = 0.5 * label_height * height_direction
    points = np.array(
        [
            center - half_width - half_height,
            center + half_width - half_height,
            center + half_width + half_height,
            center - half_width + half_height,
        ],
        dtype=np.float32,
    )
    label = pv.PolyData(points, np.array([4, 0, 1, 2, 3], dtype=np.int64))
    label.active_texture_coordinates = np.array(  # type: ignore
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    return label


def add_radius_equation_label(
    plotter: pv.Plotter,
    radius_segment: Polyline,
    dna_reference_points: Polyline,
) -> None:
    label_texture = render_equation_texture()
    label_mesh = equation_label_mesh(
        radius_segment=radius_segment,
        dna_reference_points=dna_reference_points,
        label_texture=label_texture,
        label_height=EQUATION_LABEL_HEIGHT * EQUATION_LABEL_LINE_COUNT,
        label_gap=EQUATION_LABEL_GAP,
    )
    plotter.add_mesh(label_mesh, texture=pv.Texture(label_texture))  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    base_pair_centers = generate_s_shaped_base_pair_centers()
    starts, ends, colors = dna_molecule_line_segments_from_base_pair_centers(
        base_pair_centers=base_pair_centers,
        dna_molecule_radius=DNA_MOLECULE_RADIUS,
    )
    dna_visualization = pyvista_lines_from_segments(starts, ends, colors)
    curvature_circle, radius_segment, curvature_radius = generate_curvature_circle(
        base_pair_centers
    )

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"dna_curvature_radius_{timestamp}"
        output_dir.mkdir(parents=True)
        np.save(output_dir / "base_pair_centers.npy", base_pair_centers)
        np.save(output_dir / "curvature_circle.npy", curvature_circle)
        np.save(output_dir / "curvature_radius_segment.npy", radius_segment)
        dna_mesh = dna_molecule_mesh_from_base_pair_centers(
            base_pair_centers=base_pair_centers,
            dna_molecule_radius=DNA_MOLECULE_RADIUS,
            segment_radius=DNA_SEGMENT_RADIUS,
        )
        dna_mesh.export(output_dir / "dna_curvature_radius.glb")
        dna_mesh.export(output_dir / "dna_curvature_radius.stl")
        print(output_dir)
        return 0

    plotter = pv.Plotter()
    plotter.add_mesh(  # type: ignore
        dna_visualization,
        scalars="RGBA",
        rgba=True,
        line_width=4,
        render_lines_as_tubes=True,
    )
    plotter.add_mesh(  # type: ignore
        pv.lines_from_points(curvature_circle),
        color=CURVATURE_MARKER_COLOR,
        line_width=5,
        render_lines_as_tubes=True,
    )
    plotter.add_mesh(  # type: ignore
        pv.lines_from_points(radius_segment),
        color=CURVATURE_MARKER_COLOR,
        line_width=5,
        render_lines_as_tubes=True,
    )
    add_radius_marker(plotter, radius_segment, 0.1 * curvature_radius)
    add_radius_equation_label(plotter, radius_segment, base_pair_centers)
    plotter.add_axes()  # type: ignore
    try:
        plotter.show(interactive_update=True, auto_close=False)
        while not plotter._closed:
            plotter.update(stime=10)
    except KeyboardInterrupt:
        plotter.close()
        return 130
    plotter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
