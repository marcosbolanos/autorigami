import math
import sys

import numpy as np
import numpy.typing as npt
import pyvista as pv
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from autorigami.mesh_io.tube_export import (
    ADENINE_COLOR,
    CYTOSINE_COLOR,
    DNA_BASE_PAIR_TWIST_RADIANS,
    GUANINE_COLOR,
    THYMINE_COLOR,
)
from autorigami.types import Polyline

BASE_PAIR_SPACING = 0.34
DNA_MOLECULE_RADIUS = 1.05
HELIX_CENTER_SEPARATION = 2.6
BEND_BASE_COUNT = 40
CROSSOVER_INDEX = BEND_BASE_COUNT + 21
PRE_CROSSOVER_BASE_COUNT = 10
POST_CROSSOVER_BASE_COUNT = 5
STRAIGHT_BASE_COUNT = CROSSOVER_INDEX - BEND_BASE_COUNT + POST_CROSSOVER_BASE_COUNT + 1
QUARTER_TURN_RADIANS = 0.5 * math.pi

BASE_PAIR_COMPLEMENTS = (
    (ADENINE_COLOR, THYMINE_COLOR),
    (THYMINE_COLOR, ADENINE_COLOR),
    (CYTOSINE_COLOR, GUANINE_COLOR),
    (GUANINE_COLOR, CYTOSINE_COLOR),
)
LOWER_MOLECULE_COLOR = np.array([0, 229, 255, 255], dtype=np.uint8)
UPPER_MOLECULE_COLOR = np.array([255, 122, 0, 255], dtype=np.uint8)
DIMENSION_COLOR = tuple(int(channel) for channel in CYTOSINE_COLOR[:3])
DIMENSION_LABEL_HEIGHT = 0.75
DIMENSION_LABEL_GAP = 0.35
DIMENSION_MARK_LENGTH = 0.35
DIMENSION_Y_OFFSET = 0.75


def generate_quarter_turn_then_straight_centerline() -> Polyline:
    total_base_count = BEND_BASE_COUNT + STRAIGHT_BASE_COUNT
    bend_step = QUARTER_TURN_RADIANS / float(BEND_BASE_COUNT - 1)
    interval_indices = np.arange(total_base_count - 1, dtype=np.float32)
    headings = np.minimum(interval_indices + 0.5, BEND_BASE_COUNT - 1) * bend_step
    intervals = BASE_PAIR_SPACING * np.column_stack(
        (
            np.cos(headings),
            np.sin(headings),
            np.zeros(total_base_count - 1, dtype=np.float32),
        )
    )
    centers = np.vstack(
        (np.zeros((1, 3), dtype=np.float32), np.cumsum(intervals, axis=0))
    )
    centers -= 0.5 * (centers[0] + centers[-1])
    return centers.astype(np.float32)


def polyline_frames(polyline: Polyline) -> tuple[Polyline, Polyline, Polyline]:
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    assert len(polyline) >= 2, "polyline must contain at least 2 points"

    tangents = np.gradient(polyline, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    assert np.all(tangent_norms > 0.0), (
        "polyline must not contain repeated neighboring points"
    )
    tangents = (tangents / tangent_norms).astype(np.float32)

    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normals = np.cross(tangents, reference)
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    assert np.all(normal_norms > 0.0), "polyline frame became degenerate"
    normals = (normals / normal_norms).astype(np.float32)
    binormals = np.cross(tangents, normals).astype(np.float32)
    return tangents, normals, binormals


def crossover_phase_offset(
    centerline: Polyline,
    crossover_index: int,
    global_crossover_index: int,
    neighbor_direction: npt.NDArray[np.float32],
) -> float:
    _, normals, binormals = polyline_frames(centerline)
    target_angle = math.atan2(
        float(np.dot(neighbor_direction, binormals[crossover_index])),
        float(np.dot(neighbor_direction, normals[crossover_index])),
    )
    return target_angle - float(global_crossover_index) * DNA_BASE_PAIR_TWIST_RADIANS


def strand_points(
    centerline: Polyline,
    phase_offset: float,
    first_base_index: int,
) -> tuple[Polyline, Polyline]:
    _, normals, binormals = polyline_frames(centerline)
    angles = (
        phase_offset
        + (
            first_base_index
            + np.arange(len(centerline), dtype=np.float32)
        )
        * DNA_BASE_PAIR_TWIST_RADIANS
    )
    radial_directions = (
        np.cos(angles)[:, None] * normals + np.sin(angles)[:, None] * binormals
    ).astype(np.float32)
    return (
        centerline + DNA_MOLECULE_RADIUS * radial_directions,
        centerline - DNA_MOLECULE_RADIUS * radial_directions,
    )


def add_segment(
    starts: list[npt.NDArray[np.float32]],
    ends: list[npt.NDArray[np.float32]],
    colors: list[npt.NDArray[np.uint8]],
    start: npt.NDArray[np.float32],
    end: npt.NDArray[np.float32],
    color: npt.NDArray[np.uint8],
) -> None:
    starts.append(start.astype(np.float32))
    ends.append(end.astype(np.float32))
    colors.append(color)


def dna_crossover_segments() -> tuple[
    Polyline,
    Polyline,
    npt.NDArray[np.uint8],
    Polyline,
    Polyline,
    int,
]:
    full_centerline = generate_quarter_turn_then_straight_centerline()
    display_start_index = CROSSOVER_INDEX - PRE_CROSSOVER_BASE_COUNT
    display_end_index = CROSSOVER_INDEX + POST_CROSSOVER_BASE_COUNT + 1
    assert display_start_index >= 0, "display_start_index must be non-negative"
    assert display_end_index <= len(full_centerline), (
        "display_end_index must fit inside the centerline"
    )
    local_crossover_index = CROSSOVER_INDEX - display_start_index
    centerline = full_centerline[display_start_index:display_end_index].copy()
    centerline -= 0.5 * (centerline[0] + centerline[-1])
    lower_centerline = centerline.copy()
    upper_centerline = centerline.copy()
    lower_centerline[:, 2] -= 0.5 * HELIX_CENTER_SEPARATION
    upper_centerline[:, 2] += 0.5 * HELIX_CENTER_SEPARATION

    neighbor_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    phase_offset = crossover_phase_offset(
        lower_centerline,
        local_crossover_index,
        CROSSOVER_INDEX,
        neighbor_direction,
    )
    lower_a, lower_b = strand_points(
        lower_centerline, phase_offset, display_start_index
    )
    upper_a, upper_b = strand_points(
        upper_centerline, phase_offset, display_start_index
    )

    starts: list[npt.NDArray[np.float32]] = []
    ends: list[npt.NDArray[np.float32]] = []
    colors: list[npt.NDArray[np.uint8]] = []
    rng = np.random.default_rng(0)
    complement_colors = np.array(BASE_PAIR_COMPLEMENTS, dtype=np.uint8)

    for center, strand_a, strand_b in (
        (lower_centerline, lower_a, lower_b),
        (upper_centerline, upper_a, upper_b),
    ):
        base_pair_colors = complement_colors[
            rng.integers(0, len(BASE_PAIR_COMPLEMENTS), size=len(center))
        ]
        for index in range(len(center)):
            add_segment(starts, ends, colors, strand_a[index], center[index], base_pair_colors[index, 0])
            add_segment(starts, ends, colors, center[index], strand_b[index], base_pair_colors[index, 1])

    for index in range(len(lower_centerline) - 1):
        if index != local_crossover_index:
            lower_a_color = (
                LOWER_MOLECULE_COLOR
                if index < local_crossover_index
                else UPPER_MOLECULE_COLOR
            )
            upper_b_color = (
                UPPER_MOLECULE_COLOR
                if index < local_crossover_index
                else LOWER_MOLECULE_COLOR
            )
            add_segment(starts, ends, colors, lower_a[index], lower_a[index + 1], lower_a_color)
            add_segment(starts, ends, colors, upper_b[index], upper_b[index + 1], upper_b_color)
        add_segment(starts, ends, colors, lower_b[index], lower_b[index + 1], LOWER_MOLECULE_COLOR)
        add_segment(starts, ends, colors, upper_a[index], upper_a[index + 1], UPPER_MOLECULE_COLOR)

    add_segment(
        starts,
        ends,
        colors,
        lower_a[local_crossover_index],
        upper_b[local_crossover_index + 1],
        LOWER_MOLECULE_COLOR,
    )
    add_segment(
        starts,
        ends,
        colors,
        upper_b[local_crossover_index],
        lower_a[local_crossover_index + 1],
        UPPER_MOLECULE_COLOR,
    )
    return (
        np.vstack(starts).astype(np.float32),
        np.vstack(ends).astype(np.float32),
        np.vstack(colors).astype(np.uint8),
        lower_centerline,
        upper_centerline,
        local_crossover_index,
    )


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


def add_dimension_marker(
    plotter: pv.Plotter,
    dimension_segment: Polyline,
    mark_length: float = DIMENSION_MARK_LENGTH,
) -> None:
    assert dimension_segment.shape == (2, 3), (
        "dimension_segment must have shape (2, 3)"
    )
    assert mark_length > 0.0, "mark_length must be positive"

    direction = dimension_segment[1] - dimension_segment[0]
    direction = direction / np.linalg.norm(direction)
    mark_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for endpoint in dimension_segment:
        mark = np.vstack(
            (
                endpoint - 0.5 * mark_length * mark_direction,
                endpoint + 0.5 * mark_length * mark_direction,
            )
        )
        plotter.add_mesh(  # type: ignore
            pv.lines_from_points(mark),
            color=DIMENSION_COLOR,
            line_width=7,
            render_lines_as_tubes=True,
        )


def render_dimension_texture() -> npt.NDArray[np.uint8]:
    figure = Figure(figsize=(6.0, 1.8), dpi=200)
    figure.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(figure)
    color = tuple(float(channel) / 255.0 for channel in DIMENSION_COLOR)
    figure.text(
        0.5,
        0.5,
        r"$d \approx 2.6\,\mathrm{nm}$",
        color=color,
        fontsize=76,
        ha="center",
        va="center",
    )
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    alpha = image[:, :, 3]
    occupied = np.argwhere(alpha > 0)
    assert occupied.size > 0, "dimension texture must contain rendered text"
    y_min, x_min = occupied.min(axis=0)
    y_max, x_max = occupied.max(axis=0)
    padding = 10
    y_min = max(int(y_min) - padding, 0)
    x_min = max(int(x_min) - padding, 0)
    y_max = min(int(y_max) + padding, image.shape[0] - 1)
    x_max = min(int(x_max) + padding, image.shape[1] - 1)
    return image[y_min : y_max + 1, x_min : x_max + 1].copy()


def dimension_label_mesh(
    dimension_segment: Polyline,
    label_texture: npt.NDArray[np.uint8],
    label_height: float = DIMENSION_LABEL_HEIGHT,
    label_gap: float = DIMENSION_LABEL_GAP,
) -> pv.PolyData:
    assert dimension_segment.shape == (2, 3), (
        "dimension_segment must have shape (2, 3)"
    )
    assert label_texture.ndim == 3 and label_texture.shape[2] == 4, (
        "label_texture must have shape (height, width, 4)"
    )
    assert label_height > 0.0, "label_height must be positive"
    assert label_gap > 0.0, "label_gap must be positive"

    segment_direction = dimension_segment[1] - dimension_segment[0]
    segment_direction = segment_direction / np.linalg.norm(segment_direction)
    above_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    aspect_ratio = float(label_texture.shape[1]) / float(label_texture.shape[0])
    label_width = label_height * aspect_ratio
    center = 0.5 * (dimension_segment[0] + dimension_segment[1])
    center += (label_gap + 0.5 * label_height) * above_direction
    half_width = 0.5 * label_width * segment_direction
    half_height = 0.5 * label_height * above_direction
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


def dimension_segment_after_crossover(
    lower_centerline: Polyline,
    upper_centerline: Polyline,
    local_crossover_index: int,
) -> Polyline:
    dimension_index = min(local_crossover_index + 2, len(lower_centerline) - 1)
    segment = np.vstack(
        (lower_centerline[dimension_index], upper_centerline[dimension_index])
    ).astype(np.float32)
    segment[:, 1] += DIMENSION_Y_OFFSET
    return segment


def add_distance_annotation(plotter: pv.Plotter, dimension_segment: Polyline) -> None:
    plotter.add_mesh(  # type: ignore
        pv.lines_from_points(dimension_segment),
        color=DIMENSION_COLOR,
        line_width=6,
        render_lines_as_tubes=True,
    )
    add_dimension_marker(plotter, dimension_segment)
    label_texture = render_dimension_texture()
    label_mesh = dimension_label_mesh(dimension_segment, label_texture)
    plotter.add_mesh(label_mesh, texture=pv.Texture(label_texture))  # type: ignore


def main() -> int:
    (
        starts,
        ends,
        colors,
        lower_centerline,
        upper_centerline,
        local_crossover_index,
    ) = dna_crossover_segments()
    visualization = pyvista_lines_from_segments(starts, ends, colors)
    dimension_segment = dimension_segment_after_crossover(
        lower_centerline, upper_centerline, local_crossover_index
    )

    plotter = pv.Plotter()
    plotter.add_mesh(  # type: ignore
        visualization,
        scalars="RGBA",
        rgba=True,
        line_width=4,
        render_lines_as_tubes=True,
    )
    add_distance_annotation(plotter, dimension_segment)
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
