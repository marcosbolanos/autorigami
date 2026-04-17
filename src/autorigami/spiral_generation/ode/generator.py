from __future__ import annotations

import numpy as np
import trimesh

from .integrator import integrate_tight_spiral
from .model import TightSpiralODEParams


def generate_tight_spiral_ode(
    mesh: trimesh.Trimesh,
    samples: int = 1500,
    axis_direction: np.ndarray | None = None,
    axis_origin: np.ndarray | None = None,
    world_to_nm: float = 20.0,
    target_spacing_nm: float = 2.6,
    min_curvature_radius_nm: float = 6.0,
    repulsion_strength: float = 2.0,
    repulsion_range_nm: float = 2.6,
    repulsion_lag_points: int = 8,
    tangential_speed_nm: float = 10.0,
    step_size_nm: float = 0.6,
    min_progress_fraction: float = 0.35,
    bottom_clearance_nm: float = 10.4,
    top_clearance_nm: float = 5.2,
) -> np.ndarray:
    """Generate a tight, history-aware spiral using ODE-like progressive tracing."""
    if samples < 2:
        raise ValueError("samples must be >= 2")

    axis = (
        np.array(axis_direction, dtype=float)
        if axis_direction is not None
        else np.array([0.0, 0.0, 1.0], dtype=float)
    )
    origin = (
        np.array(axis_origin, dtype=float)
        if axis_origin is not None
        else np.array(mesh.bounding_box.centroid, dtype=float)
    )

    vertices = np.asarray(mesh.vertices)
    axis_unit = axis / np.linalg.norm(axis)
    axis_pos = (vertices - origin) @ axis_unit

    params = TightSpiralODEParams(
        world_to_nm=world_to_nm,
        target_spacing_nm=target_spacing_nm,
        min_curvature_radius_nm=min_curvature_radius_nm,
        repulsion_strength=repulsion_strength,
        repulsion_range_nm=repulsion_range_nm,
        repulsion_lag_points=repulsion_lag_points,
        tangential_speed_nm=tangential_speed_nm,
        step_size_nm=step_size_nm,
        min_progress_fraction=min_progress_fraction,
        bottom_clearance_nm=bottom_clearance_nm,
        top_clearance_nm=top_clearance_nm,
    )
    start_axis_pos = float(np.min(axis_pos)) + params.bottom_clearance_world
    start_vertex = vertices[int(np.argmin(np.abs(axis_pos - start_axis_pos)))]

    return integrate_tight_spiral(
        mesh=mesh,
        start_point=start_vertex,
        axis_direction=axis,
        axis_origin=origin,
        params=params,
        num_points=samples,
    )
