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
from trimesh.visual.color import ColorVisuals

from autorigami.mesh_io import (
    dna_molecule_line_segments_from_base_pair_centers,
    dna_molecule_mesh_from_base_pair_centers,
)
from autorigami.types import Polyline

TURN_ANGLE = 0.057
DISTANCE_BETWEEN_BASE_PAIRS = 0.34
DNA_MOLECULE_RADIUS = 1.05
DNA_SEGMENT_RADIUS = 0.05
REVERSE_CURVE_BASE_COUNT = 15
NON_REVERSE_END_TRIM_BASE_COUNT = 5

GREEN_STRAND_COLOR = np.array([24, 116, 68, 255], dtype=np.uint8)
GREEN_BASE_COLORS = (
    np.array([64, 170, 92, 255], dtype=np.uint8),
    np.array([137, 214, 145, 255], dtype=np.uint8),
)
RED_STRAND_COLOR = np.array([174, 35, 48, 255], dtype=np.uint8)
RED_BASE_COLORS = (
    np.array([220, 68, 65, 255], dtype=np.uint8),
    np.array([255, 143, 123, 255], dtype=np.uint8),
)
SAFE_LABEL_HEIGHT = 0.85
DISTANCE_LABEL_HEIGHT = 0.65
LABEL_GAP = 0.8
DISTANCE_LABEL_GAP = 0.3


def generate_self_intersecting_base_pair_centers(
    turn_angle: float = TURN_ANGLE,
    distance_between_base_pairs: float = DISTANCE_BETWEEN_BASE_PAIRS,
    dna_molecule_radius: float = DNA_MOLECULE_RADIUS,
    reverse_curve_base_count: int = REVERSE_CURVE_BASE_COUNT,
    non_reverse_end_trim_base_count: int = NON_REVERSE_END_TRIM_BASE_COUNT,
) -> Polyline:
    assert 0.0 < turn_angle < math.pi, "turn_angle must be between 0 and pi"
    assert distance_between_base_pairs > 0.0, (
        "distance_between_base_pairs must be positive"
    )
    assert dna_molecule_radius > 0.0, "dna_molecule_radius must be positive"
    assert reverse_curve_base_count >= 2, (
        "reverse_curve_base_count must be at least 2"
    )
    assert non_reverse_end_trim_base_count >= 0, (
        "non_reverse_end_trim_base_count must be non-negative"
    )

    maximum_interval_count = math.ceil(2.0 * math.pi / turn_angle)
    interval_indices = np.arange(maximum_interval_count, dtype=np.float32)
    headings = (interval_indices + 0.5) * np.float32(turn_angle)
    intervals = distance_between_base_pairs * np.column_stack(
        (
            np.cos(headings),
            np.sin(headings),
            np.zeros(maximum_interval_count, dtype=np.float32),
        )
    )
    circular_centers = np.vstack(
        (np.zeros((1, 3), dtype=np.float32), np.cumsum(intervals, axis=0))
    )
    minimum_index_separation = math.ceil(
        2.0 * dna_molecule_radius / distance_between_base_pairs
    )
    intersection_end_index = next(
        end_index
        for end_index in range(
            minimum_index_separation + 1, len(circular_centers)
        )
        if np.any(
            np.linalg.norm(
                circular_centers[: end_index - minimum_index_separation]
                - circular_centers[end_index],
                axis=1,
            )
            <= 2.0 * dna_molecule_radius
        )
    )
    circular_base_pair_count = intersection_end_index + 1
    assert non_reverse_end_trim_base_count < circular_base_pair_count, (
        "non_reverse_end_trim_base_count must leave part of the circular curve"
    )
    base_pair_count = (
        reverse_curve_base_count
        + circular_base_pair_count
        - non_reverse_end_trim_base_count
    )
    turn_signs = np.ones(base_pair_count, dtype=np.float32)
    turn_signs[:reverse_curve_base_count] = -1.0
    headings = np.cumsum(turn_signs * np.float32(turn_angle), dtype=np.float32)
    interval_headings = 0.5 * (headings[:-1] + headings[1:])
    intervals = distance_between_base_pairs * np.column_stack(
        (
            np.cos(interval_headings),
            np.sin(interval_headings),
            np.zeros(base_pair_count - 1, dtype=np.float32),
        )
    )
    centers = np.vstack(
        (np.zeros((1, 3), dtype=np.float32), np.cumsum(intervals, axis=0))
    )
    centers -= centers.mean(axis=0)
    return centers.astype(np.float32)


def self_intersection_mask(
    base_pair_centers: Polyline,
    dna_molecule_radius: float = DNA_MOLECULE_RADIUS,
    distance_between_base_pairs: float = DISTANCE_BETWEEN_BASE_PAIRS,
) -> npt.NDArray[np.bool_]:
    assert base_pair_centers.ndim == 2 and base_pair_centers.shape[1] == 3, (
        "base_pair_centers must have shape (n, 3)"
    )
    assert len(base_pair_centers) >= 3, (
        "base_pair_centers must contain at least 3 points"
    )
    assert dna_molecule_radius > 0.0, "dna_molecule_radius must be positive"
    assert distance_between_base_pairs > 0.0, (
        "distance_between_base_pairs must be positive"
    )

    minimum_index_separation = math.ceil(
        2.0 * dna_molecule_radius / distance_between_base_pairs
    )
    index_distances = np.abs(
        np.subtract.outer(
            np.arange(len(base_pair_centers)),
            np.arange(len(base_pair_centers)),
        )
    )
    point_distances = np.linalg.norm(
        base_pair_centers[:, None, :] - base_pair_centers[None, :, :], axis=2
    )
    intersecting_pairs = (index_distances > minimum_index_separation) & (
        point_distances <= 2.0 * dna_molecule_radius
    )
    mask = np.any(intersecting_pairs, axis=1)
    assert np.any(mask), "centerline must continue far enough to intersect itself"
    return mask


def recolor_dna_segments(
    colors: npt.NDArray[np.uint8],
    intersecting_base_pairs: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    base_pair_count = len(intersecting_base_pairs)
    strand_segment_count = base_pair_count - 1
    expected_segment_count = 2 * strand_segment_count + 2 * base_pair_count
    assert colors.shape == (expected_segment_count, 4), (
        "colors must match the DNA segment layout"
    )
    assert intersecting_base_pairs.shape == (base_pair_count,), (
        "intersecting_base_pairs must have shape (base_pair_count,)"
    )

    recolored = colors.copy()
    strand_intersections = (
        intersecting_base_pairs[:-1] | intersecting_base_pairs[1:]
    )
    for strand_index in range(2):
        start = strand_index * strand_segment_count
        stop = start + strand_segment_count
        recolored[start:stop] = GREEN_STRAND_COLOR
        recolored[start:stop][strand_intersections] = RED_STRAND_COLOR

    base_segment_start = 2 * strand_segment_count
    first_base_slice = slice(base_segment_start, base_segment_start + base_pair_count)
    second_base_slice = slice(
        base_segment_start + base_pair_count,
        base_segment_start + 2 * base_pair_count,
    )
    recolored[first_base_slice] = GREEN_BASE_COLORS[0]
    recolored[second_base_slice] = GREEN_BASE_COLORS[1]
    recolored[first_base_slice][intersecting_base_pairs] = RED_BASE_COLORS[0]
    recolored[second_base_slice][intersecting_base_pairs] = RED_BASE_COLORS[1]
    return recolored


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


def render_text_texture(
    text: str,
    color: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    assert text, "text must not be empty"
    assert color.shape == (4,), "color must have shape (4,)"

    figure = Figure(figsize=(9.0, 1.7), dpi=200)
    figure.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(figure)
    figure.text(
        0.5,
        0.5,
        text,
        color=tuple(float(channel) / 255.0 for channel in color[:3]),
        fontsize=56,
        ha="center",
        va="center",
    )
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    occupied = np.argwhere(image[:, :, 3] > 0)
    assert occupied.size > 0, "text texture must contain rendered text"
    y_min, x_min = occupied.min(axis=0)
    y_max, x_max = occupied.max(axis=0)
    padding = 10
    return image[
        max(int(y_min) - padding, 0) : min(int(y_max) + padding + 1, image.shape[0]),
        max(int(x_min) - padding, 0) : min(int(x_max) + padding + 1, image.shape[1]),
    ].copy()


def text_label_mesh(
    label_texture: npt.NDArray[np.uint8],
    center: npt.NDArray[np.float32],
    label_height: float,
) -> pv.PolyData:
    assert label_texture.ndim == 3 and label_texture.shape[2] == 4, (
        "label_texture must have shape (height, width, 4)"
    )
    assert center.shape == (3,), "center must have shape (3,)"
    assert label_height > 0.0, "label_height must be positive"

    aspect_ratio = float(label_texture.shape[1]) / float(label_texture.shape[0])
    label_width = label_height * aspect_ratio
    half_width = np.array([0.5 * label_width, 0.0, 0.0], dtype=np.float32)
    half_height = np.array([0.0, 0.5 * label_height, 0.0], dtype=np.float32)
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
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    return label


def add_area_labels(
    plotter: pv.Plotter,
    base_pair_centers: Polyline,
    intersecting_base_pairs: npt.NDArray[np.bool_],
) -> None:
    assert intersecting_base_pairs.shape == (len(base_pair_centers),), (
        "intersecting_base_pairs must match base_pair_centers"
    )

    safe_texture = render_text_texture(
        r"$\mathrm{safe\ area}$",
        GREEN_STRAND_COLOR,
    )
    distance_texture = render_text_texture(
        r"$\mathrm{must\ check\ distance}$",
        RED_STRAND_COLOR,
    )
    safe_center = np.array(
        [
            0.0,
            float(np.max(base_pair_centers[:, 1]))
            + DNA_MOLECULE_RADIUS
            + LABEL_GAP,
            0.2,
        ],
        dtype=np.float32,
    )
    red_centers = base_pair_centers[intersecting_base_pairs]
    distance_label_width = DISTANCE_LABEL_HEIGHT * (
        float(distance_texture.shape[1]) / float(distance_texture.shape[0])
    )
    distance_center = np.array(
        [
            float(np.max(red_centers[:, 0])) - 0.5 * distance_label_width,
            float(np.min(red_centers[:, 1]))
            - DNA_MOLECULE_RADIUS
            - DISTANCE_LABEL_GAP
            - 0.5 * DISTANCE_LABEL_HEIGHT,
            0.2,
        ],
        dtype=np.float32,
    )
    plotter.add_mesh(  # type: ignore
        text_label_mesh(safe_texture, safe_center, SAFE_LABEL_HEIGHT),
        texture=pv.Texture(safe_texture),  # type: ignore
    )
    plotter.add_mesh(  # type: ignore
        text_label_mesh(distance_texture, distance_center, DISTANCE_LABEL_HEIGHT),
        texture=pv.Texture(distance_texture),  # type: ignore
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    base_pair_centers = generate_self_intersecting_base_pair_centers()
    intersecting_base_pairs = self_intersection_mask(base_pair_centers)
    starts, ends, original_colors = (
        dna_molecule_line_segments_from_base_pair_centers(
            base_pair_centers=base_pair_centers,
            dna_molecule_radius=DNA_MOLECULE_RADIUS,
        )
    )
    colors = recolor_dna_segments(original_colors, intersecting_base_pairs)
    dna_visualization = pyvista_lines_from_segments(starts, ends, colors)

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"self_intersection_curvature_bound_{timestamp}"
        output_dir.mkdir(parents=True)
        np.save(output_dir / "base_pair_centers.npy", base_pair_centers)
        np.save(output_dir / "intersecting_base_pairs.npy", intersecting_base_pairs)
        dna_mesh = dna_molecule_mesh_from_base_pair_centers(
            base_pair_centers=base_pair_centers,
            dna_molecule_radius=DNA_MOLECULE_RADIUS,
            segment_radius=DNA_SEGMENT_RADIUS,
        )
        dna_mesh.visual = ColorVisuals(
            mesh=dna_mesh,
            vertex_colors=np.repeat(colors, 8, axis=0),
        )
        dna_mesh.export(output_dir / "self_intersection_curvature_bound.glb")
        dna_mesh.export(output_dir / "self_intersection_curvature_bound.stl")
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
    add_area_labels(plotter, base_pair_centers, intersecting_base_pairs)
    plotter.camera_position = [
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    plotter.reset_camera()  # type: ignore
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
