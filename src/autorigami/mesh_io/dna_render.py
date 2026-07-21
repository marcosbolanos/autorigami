from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyvista as pv
from trimesh.visual.color import ColorVisuals

from autorigami.geometry.validation import PolylineViolationMasks
from autorigami.mesh_io.tube_export import dna_molecule_mesh_from_base_pair_centers
from autorigami.types import Polyline

CameraView = Literal["xy", "xz", "yz", "isometric"]


def render_dna_molecule_png(
    base_pair_centers: Polyline,
    path: Path,
    *,
    dna_molecule_radius: float = 1.05,
    segment_radius: float = 0.05,
    curvature_marker_radius: float = 3.2,
    violation_masks: PolylineViolationMasks | None = None,
    background_color: str = "white",
    window_size: tuple[int, int] = (1000, 760),
    camera_view: CameraView = "xz",
    camera_zoom: float = 1.45,
) -> Path:
    """Render DNA, optionally highlighting standard validation masks.

    Separation-violating centerline edges receive an opaque red tube. A
    curvature-violating vertex receives a semi-transparent red sphere with
    the requested physical radius.
    """
    assert base_pair_centers.ndim == 2 and base_pair_centers.shape[1] == 3
    assert len(base_pair_centers) >= 2
    assert dna_molecule_radius > 0.0
    assert segment_radius > 0.0
    assert curvature_marker_radius > 0.0
    assert window_size[0] > 0 and window_size[1] > 0
    assert camera_zoom > 0.0
    if violation_masks is not None:
        assert violation_masks.separation_edges.shape == (len(base_pair_centers) - 1,)
        assert violation_masks.curvature_vertices.shape == (len(base_pair_centers),)

    dna_mesh = dna_molecule_mesh_from_base_pair_centers(
        base_pair_centers,
        dna_molecule_radius=dna_molecule_radius,
        segment_radius=segment_radius,
    )
    dna_polydata = pv.wrap(dna_mesh)
    dna_visual = cast(ColorVisuals, dna_mesh.visual)
    dna_polydata.point_data["RGBA"] = np.asarray(
        dna_visual.vertex_colors, dtype=np.uint8
    )
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(background_color)  # type: ignore
    plotter.add_mesh(  # type: ignore
        dna_polydata,
        scalars="RGBA",
        rgba=True,
        lighting=True,
    )
    if violation_masks is not None:
        _add_violation_overlays(
            plotter,
            base_pair_centers=base_pair_centers,
            violation_masks=violation_masks,
            separation_marker_radius=dna_molecule_radius + 0.08,
            curvature_marker_radius=curvature_marker_radius,
        )

    if camera_view == "xy":
        plotter.view_xy()  # type: ignore
    elif camera_view == "xz":
        plotter.view_xz()  # type: ignore
    elif camera_view == "yz":
        plotter.view_yz()  # type: ignore
    else:
        plotter.view_isometric()  # type: ignore
    plotter.enable_parallel_projection()  # type: ignore
    plotter.reset_camera()  # type: ignore
    plotter.camera.zoom(camera_zoom)
    plotter.screenshot(path)
    plotter.close()
    return path


def _add_violation_overlays(
    plotter: pv.Plotter,
    *,
    base_pair_centers: Polyline,
    violation_masks: PolylineViolationMasks,
    separation_marker_radius: float,
    curvature_marker_radius: float,
) -> None:
    violating_edge_indices = np.flatnonzero(violation_masks.separation_edges)
    if len(violating_edge_indices) > 0:
        starts = base_pair_centers[violating_edge_indices]
        ends = base_pair_centers[violating_edge_indices + 1]
        segment_points = np.stack((starts, ends), axis=1).reshape((-1, 3))
        segment_lines = np.column_stack(
            (
                np.full(len(violating_edge_indices), 2, dtype=np.int64),
                2 * np.arange(len(violating_edge_indices), dtype=np.int64),
                2 * np.arange(len(violating_edge_indices), dtype=np.int64) + 1,
            )
        ).reshape(-1)
        edge_lines = pv.PolyData(segment_points, lines=segment_lines)
        edge_tubes = edge_lines.tube(
            radius=separation_marker_radius,
            n_sides=10,
            capping=True,
        )
        plotter.add_mesh(edge_tubes, color="red", lighting=True)  # type: ignore

    violating_vertices = base_pair_centers[violation_masks.curvature_vertices]
    if len(violating_vertices) > 0:
        marker_centers = pv.PolyData(violating_vertices)
        marker = pv.Sphere(
            radius=curvature_marker_radius,
            theta_resolution=12,
            phi_resolution=12,
        )
        spheres = cast(
            pv.PolyData,
            marker_centers.glyph(geom=marker, orient=False, scale=False),
        )
        plotter.add_mesh(  # type: ignore
            spheres,
            color="red",
            opacity=0.32,
            lighting=False,
        )
