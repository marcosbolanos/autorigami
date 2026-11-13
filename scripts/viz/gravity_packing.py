"""Render the V1 three-way tube before and after gravity packing.

The figure has one column per state (before, after) and one row per camera
(side, isometric). Both columns of a row use the same camera.
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from autorigami.geometry.curvature import DEFAULT_EDGE_LENGTH_NM
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.geometry.validation import validate_polyline
from autorigami.packing import pack_under_gravity
from autorigami.spiral_generation.three_way_tube_v1 import generate_full_spiral
from autorigami.types import Polyline

DNA_RADIUS = 1.0
TUBE_COLOR = "#5b8fa3"
NEIGHBOR_DISTANCE = 2.8
IGNORED_ADJACENT_VERTICES = 100
PANEL_SIZE = (1100, 900)


def neighbor_fraction(points: Polyline) -> float:
    """Fraction of vertices that have another turn within NEIGHBOR_DISTANCE."""
    pairs = cKDTree(points).query_pairs(NEIGHBOR_DISTANCE, output_type="ndarray")
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > IGNORED_ADJACENT_VERTICES]
    has_neighbor = np.zeros(len(points), dtype=np.bool_)
    has_neighbor[pairs.ravel()] = True
    return float(has_neighbor.mean())


def report(name: str, points: Polyline) -> None:
    validation = validate_polyline(points)
    print(
        f"{name}: valid={validation.valid} "
        f"curvature_violations={validation.curvature_violation_count} "
        f"separation_violations={validation.separation_violation_count} "
        f"vertices={len(points)} "
        f"length_nm={DEFAULT_EDGE_LENGTH_NM * (len(points) - 1):.0f} "
        f"height_nm={np.ptp(points[:, 2]):.1f} "
        f"neighbor_fraction={neighbor_fraction(points):.3f}"
    )


def render(before: Polyline, after: Polyline, output: Path) -> None:
    bounds = np.column_stack(
        (
            np.minimum(before.min(axis=0), after.min(axis=0)) - DNA_RADIUS,
            np.maximum(before.max(axis=0), after.max(axis=0)) + DNA_RADIUS,
        )
    ).ravel()
    plotter = pv.Plotter(
        off_screen=True,
        shape=(2, 2),
        window_size=[2 * PANEL_SIZE[0], 2 * PANEL_SIZE[1]],
        border=False,
    )
    plotter.set_background("white")  # type: ignore
    for column, (title, points) in enumerate((("Before", before), ("After", after))):
        tube = pv.lines_from_points(points).tube(radius=DNA_RADIUS, n_sides=12)
        for row in range(2):
            plotter.subplot(row, column)
            plotter.add_mesh(tube, color=TUBE_COLOR, smooth_shading=True)  # type: ignore
            # The invisible box gives both columns the same camera.
            plotter.add_mesh(pv.Box(bounds), opacity=0.0)  # type: ignore
            if row == 0:
                plotter.add_text(title, position="upper_edge", font_size=22, color="black")
                plotter.view_xz()  # type: ignore
            else:
                plotter.view_isometric()  # type: ignore
            plotter.enable_parallel_projection()  # type: ignore
            plotter.camera.zoom(1.25)
    plotter.screenshot(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    before = reparametrize_arc_length(generate_full_spiral(), DEFAULT_EDGE_LENGTH_NM)
    result = pack_under_gravity(generate_full_spiral())
    report("before", before)
    report("after", result.points)
    print(f"maximum_angle_deg={np.degrees(result.validation.maximum_angle):.3f}")
    render(before, result.points, args.output)


if __name__ == "__main__":
    main()
