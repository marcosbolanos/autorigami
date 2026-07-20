import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from trimesh.visual.color import ColorVisuals

from autorigami.mesh_io import (
    dna_molecule_line_segments_from_base_pair_centers,
    dna_molecule_mesh_from_base_pair_centers,
)
from autorigami.types import Polyline

BASE_PAIR_SPACING = 0.34
DNA_MOLECULE_RADIUS = 1.05
DNA_SEGMENT_RADIUS = 0.05
OUTER_BASE_COUNT = 12
LOOP_BASE_COUNT = 4
MAX_OUTER_TURN_DEGREES = 3.25
RANDOM_SEED = 0
NONLOCAL_BASE_PAIR_SEPARATION = 10
PLOT_LINE_COLOR = "#00c6d7"
PLOT_LINE_WIDTH = 3.0
PLOT_FONT_SIZE = 16.0
PLOT_TITLE_FONT_SIZE = 19.2
UNSAFE_METRIC_THRESHOLD = 2.6
UNSAFE_ZONE_COLOR = "#d62728"
UNSAFE_ZONE_ALPHA = 0.18
FIGURE_CROP_PADDING_INCHES = 0.08

PURPLE_STRAND_COLOR = np.array([88, 64, 205, 255], dtype=np.uint8)
PURPLE_BASE_COLORS = (
    np.array([128, 58, 180, 255], dtype=np.uint8),
    np.array([174, 118, 230, 255], dtype=np.uint8),
)
RED_STRAND_COLOR = np.array([174, 35, 48, 255], dtype=np.uint8)
RED_BASE_COLORS = (
    np.array([220, 68, 65, 255], dtype=np.uint8),
    np.array([255, 143, 123, 255], dtype=np.uint8),
)


def oscillating_turn_angles(
    turn_count: int,
    maximum_turn_degrees: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    assert turn_count >= 1, "turn_count must be at least 1"
    assert maximum_turn_degrees > 0.0, "maximum_turn_degrees must be positive"

    magnitudes = rng.uniform(0.0, maximum_turn_degrees, size=turn_count)
    signs = np.where(np.arange(turn_count) % 2 == 0, 1.0, -1.0)
    return np.radians(magnitudes * signs).astype(np.float32)


def generate_loop_base_pair_centers(
    outer_base_count: int = OUTER_BASE_COUNT,
    loop_base_count: int = LOOP_BASE_COUNT,
    base_pair_spacing: float = BASE_PAIR_SPACING,
    maximum_outer_turn_degrees: float = MAX_OUTER_TURN_DEGREES,
    random_seed: int = RANDOM_SEED,
) -> tuple[Polyline, npt.NDArray[np.bool_]]:
    assert outer_base_count >= 2, "outer_base_count must be at least 2"
    assert loop_base_count >= 4, "loop_base_count must be at least 4"
    assert base_pair_spacing > 0.0, "base_pair_spacing must be positive"
    assert 0.0 < maximum_outer_turn_degrees < 180.0, (
        "maximum_outer_turn_degrees must be between 0 and 180"
    )

    rng = np.random.default_rng(random_seed)
    outer_interval_count = outer_base_count - 1
    pre_turns = oscillating_turn_angles(
        outer_interval_count,
        maximum_outer_turn_degrees,
        rng,
    )
    pre_headings = np.cumsum(pre_turns, dtype=np.float32)

    loop_interval_count = loop_base_count - 1
    loop_turn_angle = 2.0 * math.pi / float(loop_interval_count)
    entry_heading = float(pre_headings[-1])
    loop_headings = entry_heading + loop_turn_angle * np.arange(
        loop_interval_count, dtype=np.float32
    )

    post_turns = oscillating_turn_angles(
        outer_interval_count,
        maximum_outer_turn_degrees,
        rng,
    )
    post_headings = (
        entry_heading + 2.0 * math.pi + np.cumsum(post_turns, dtype=np.float32)
    )
    headings = np.concatenate((pre_headings, loop_headings, post_headings))
    intervals = base_pair_spacing * np.column_stack(
        (
            np.cos(headings),
            np.sin(headings),
            np.zeros(len(headings), dtype=np.float32),
        )
    )
    centers = np.vstack(
        (np.zeros((1, 3), dtype=np.float32), np.cumsum(intervals, axis=0))
    )
    centers -= 0.5 * (centers[0] + centers[-1])

    loop_start_index = outer_base_count - 1
    loop_stop_index = loop_start_index + loop_base_count
    loop_mask = np.zeros(len(centers), dtype=np.bool_)
    loop_mask[loop_start_index:loop_stop_index] = True
    assert int(np.sum(loop_mask)) == loop_base_count, (
        "loop_mask must select exactly loop_base_count bases"
    )
    return centers.astype(np.float32), loop_mask


def recolor_dna_segments(
    colors: npt.NDArray[np.uint8],
    loop_base_pairs: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    base_pair_count = len(loop_base_pairs)
    strand_segment_count = base_pair_count - 1
    expected_segment_count = 2 * strand_segment_count + 2 * base_pair_count
    assert colors.shape == (expected_segment_count, 4), (
        "colors must match the DNA segment layout"
    )
    assert loop_base_pairs.shape == (base_pair_count,), (
        "loop_base_pairs must have shape (base_pair_count,)"
    )

    recolored = colors.copy()
    loop_strand_segments = loop_base_pairs[:-1] | loop_base_pairs[1:]
    for strand_index in range(2):
        start = strand_index * strand_segment_count
        stop = start + strand_segment_count
        recolored[start:stop] = PURPLE_STRAND_COLOR
        recolored[start:stop][loop_strand_segments] = RED_STRAND_COLOR

    base_segment_start = 2 * strand_segment_count
    first_base_slice = slice(base_segment_start, base_segment_start + base_pair_count)
    second_base_slice = slice(
        base_segment_start + base_pair_count,
        base_segment_start + 2 * base_pair_count,
    )
    recolored[first_base_slice] = PURPLE_BASE_COLORS[0]
    recolored[second_base_slice] = PURPLE_BASE_COLORS[1]
    recolored[first_base_slice][loop_base_pairs] = RED_BASE_COLORS[0]
    recolored[second_base_slice][loop_base_pairs] = RED_BASE_COLORS[1]
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


def nonlocal_separation(
    base_pair_centers: Polyline,
    excluded_base_pair_distance: int = NONLOCAL_BASE_PAIR_SEPARATION,
) -> npt.NDArray[np.float32]:
    assert base_pair_centers.ndim == 2 and base_pair_centers.shape[1] == 3, (
        "base_pair_centers must have shape (n, 3)"
    )
    assert excluded_base_pair_distance >= 1, (
        "excluded_base_pair_distance must be at least 1"
    )
    assert len(base_pair_centers) > 2 * excluded_base_pair_distance + 1, (
        "every base pair must have a nonlocal comparison point"
    )

    indices = np.arange(len(base_pair_centers))
    valid_pairs = np.abs(np.subtract.outer(indices, indices)) > (
        excluded_base_pair_distance
    )
    distances = np.linalg.norm(
        base_pair_centers[:, None, :] - base_pair_centers[None, :, :],
        axis=2,
    )
    distances[~valid_pairs] = np.inf
    minimum_distances = np.min(distances, axis=1)
    assert np.all(np.isfinite(minimum_distances)), (
        "every base pair must have a finite nonlocal separation"
    )
    return minimum_distances.astype(np.float32)


def tangent_sphere_radius(
    base_pair_centers: Polyline,
    excluded_base_pair_distance: int = NONLOCAL_BASE_PAIR_SEPARATION,
) -> npt.NDArray[np.float32]:
    assert base_pair_centers.ndim == 2 and base_pair_centers.shape[1] == 3, (
        "base_pair_centers must have shape (n, 3)"
    )
    assert excluded_base_pair_distance >= 1, (
        "excluded_base_pair_distance must be at least 1"
    )
    assert len(base_pair_centers) > 2 * excluded_base_pair_distance + 1, (
        "every base pair must have a nonlocal comparison point"
    )

    tangents = np.gradient(base_pair_centers, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    assert np.all(tangent_norms > 0.0), "centerline tangents must be nonzero"
    tangents = tangents / tangent_norms
    displacements = base_pair_centers[:, None, :] - base_pair_centers[None, :, :]
    squared_distances = np.sum(displacements * displacements, axis=2)
    tangent_projections = np.sum(displacements * tangents[:, None, :], axis=2)
    squared_normal_distances = np.maximum(
        squared_distances - tangent_projections * tangent_projections,
        0.0,
    )

    indices = np.arange(len(base_pair_centers))
    valid_pairs = np.abs(np.subtract.outer(indices, indices)) > (
        excluded_base_pair_distance
    )
    assert not np.any(valid_pairs & (squared_distances == 0.0)), (
        "nonlocal tangent-sphere samples must be distinct"
    )
    denominators = 2.0 * np.sqrt(squared_normal_distances)
    radii = np.full(squared_distances.shape, np.inf, dtype=np.float32)
    np.divide(
        squared_distances,
        denominators,
        out=radii,
        where=valid_pairs & (denominators > 0.0),
    )
    minimum_radii = np.min(radii, axis=1)
    assert np.all(np.isfinite(minimum_radii)), (
        "every base pair must have a finite tangent sphere radius"
    )
    return minimum_radii.astype(np.float32)


def diagnostic_figure(base_pair_centers: Polyline) -> Figure:
    positions = np.arange(len(base_pair_centers))
    separation = nonlocal_separation(base_pair_centers)
    sphere_radius = tangent_sphere_radius(base_pair_centers)
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].plot(
        positions,
        separation,
        color=PLOT_LINE_COLOR,
        linewidth=PLOT_LINE_WIDTH,
    )
    axes[0].set_title(
        "nonlocal separation (>10bp)",
        fontsize=PLOT_TITLE_FONT_SIZE,
    )
    axes[1].plot(
        positions,
        sphere_radius,
        color=PLOT_LINE_COLOR,
        linewidth=PLOT_LINE_WIDTH,
    )
    axes[1].set_title("tangent sphere radius", fontsize=PLOT_TITLE_FONT_SIZE)
    for axis in axes:
        axis.axhspan(
            0.0,
            UNSAFE_METRIC_THRESHOLD,
            color=UNSAFE_ZONE_COLOR,
            alpha=UNSAFE_ZONE_ALPHA,
            linewidth=0.0,
        )
        axis.axhline(
            UNSAFE_METRIC_THRESHOLD,
            color="black",
            linestyle=":",
            linewidth=PLOT_LINE_WIDTH,
        )
        axis.set_ylim(bottom=0.0)
        axis.tick_params(axis="both", labelsize=PLOT_FONT_SIZE)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument(
        "--loop-base-count",
        type=int,
        default=LOOP_BASE_COUNT,
        help="number of bases forming the tight loop",
    )
    args = parser.parse_args()

    base_pair_centers, loop_base_pairs = generate_loop_base_pair_centers(
        loop_base_count=args.loop_base_count
    )
    starts, ends, original_colors = (
        dna_molecule_line_segments_from_base_pair_centers(
            base_pair_centers=base_pair_centers,
            dna_molecule_radius=DNA_MOLECULE_RADIUS,
        )
    )
    colors = recolor_dna_segments(original_colors, loop_base_pairs)
    metrics_figure = diagnostic_figure(base_pair_centers)

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"loop_{timestamp}"
        output_dir.mkdir(parents=True)
        np.save(output_dir / "base_pair_centers.npy", base_pair_centers)
        np.save(output_dir / "loop_base_pairs.npy", loop_base_pairs)
        dna_mesh = dna_molecule_mesh_from_base_pair_centers(
            base_pair_centers=base_pair_centers,
            dna_molecule_radius=DNA_MOLECULE_RADIUS,
            segment_radius=DNA_SEGMENT_RADIUS,
        )
        dna_mesh.visual = ColorVisuals(
            mesh=dna_mesh,
            vertex_colors=np.repeat(colors, 8, axis=0),
        )
        dna_mesh.export(output_dir / "loop.glb")
        dna_mesh.export(output_dir / "loop.stl")
        metrics_figure.savefig(
            output_dir / "loop_metrics.png",
            dpi=200,
            bbox_inches="tight",
            pad_inches=FIGURE_CROP_PADDING_INCHES,
        )
        plt.close(metrics_figure)
        print(output_dir)
        return 0

    plotter = pv.Plotter()
    plotter.add_mesh(  # type: ignore
        pyvista_lines_from_segments(starts, ends, colors),
        scalars="RGBA",
        rgba=True,
        line_width=4,
        render_lines_as_tubes=True,
    )
    plotter.view_xy()  # type: ignore
    metrics_figure.show()
    try:
        plotter.show(interactive_update=True, auto_close=False)
        while not plotter._closed:
            plotter.update(stime=10)
    except KeyboardInterrupt:
        plotter.close()
        return 130
    plotter.close()
    plt.close(metrics_figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
