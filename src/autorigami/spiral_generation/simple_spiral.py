"""Simple spiral generation on a mesh surface."""

from __future__ import annotations

import numpy as np
import trimesh


def generate_spiral_on_surface(
    mesh: trimesh.Trimesh,
    turns: float = 10.0,
    samples: int = 1500,
    radial_scale: float = 0.45,
    axis_direction: np.ndarray | None = None,
    axis_origin: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a spiral polyline that is projected onto a mesh surface.

    The seed spiral is built around a configurable axis, then each sample is
    projected to the closest surface point.

    Args:
        mesh: Input mesh.
        turns: Number of spiral turns from bottom to top.
        samples: Number of polyline samples.
        radial_scale: Fraction of mesh radius around the axis used for seed radius.
        axis_direction: 3D axis direction vector.
        axis_origin: 3D point on the axis.

    Returns:
        Polyline as a NumPy array with shape (samples, 3).
    """
    if samples < 2:
        raise ValueError("samples must be >= 2")
    if turns <= 0:
        raise ValueError("turns must be > 0")
    if not (0 < radial_scale <= 1.0):
        raise ValueError("radial_scale must be in (0, 1]")

    axis = (
        np.array(axis_direction, dtype=float)
        if axis_direction is not None
        else np.array([0.0, 0.0, 1.0], dtype=float)
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("axis_direction must be non-zero.")
    axis /= axis_norm

    origin = (
        np.array(axis_origin, dtype=float)
        if axis_origin is not None
        else np.array(mesh.bounding_box.centroid, dtype=float)
    )
    if origin.shape != (3,):
        raise ValueError("axis_origin must be a 3D vector.")

    vertices = np.asarray(mesh.vertices)
    rel = vertices - origin
    axis_pos = rel @ axis
    if axis_pos.size == 0:
        raise ValueError("Mesh has no vertices.")

    axis_min = float(axis_pos.min())
    axis_max = float(axis_pos.max())

    perp = rel - np.outer(axis_pos, axis)
    mesh_radius = float(np.linalg.norm(perp, axis=1).max())
    if mesh_radius == 0:
        raise ValueError("Degenerate mesh around the chosen axis.")

    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    t = np.linspace(0.0, 1.0, samples)
    theta = t * turns * 2.0 * np.pi
    axis_points = origin + np.outer(np.linspace(axis_min, axis_max, samples), axis)
    radius = radial_scale * mesh_radius
    circle_offsets = (
        radius * np.cos(theta)[:, None] * u[None, :]
        + radius * np.sin(theta)[:, None] * v[None, :]
    )
    seed = axis_points + circle_offsets

    projected, _, _ = mesh.nearest.on_surface(seed)
    return projected
