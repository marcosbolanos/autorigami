from __future__ import annotations

import numpy as np
import trimesh

from .geometry import (
    clamp_direction_by_curvature,
    estimate_circumference,
    make_tangent_frame,
    normalize,
    project_to_tangent_plane,
)
from .model import TightSpiralODEParams
from .repulsion import repulsive_potential_gradient


def _nearest_on_surface(mesh: trimesh.Trimesh, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected, _, face_idx = mesh.nearest.on_surface(point[None, :])
    idx = int(face_idx[0])
    normal = np.array(mesh.face_normals[idx], dtype=float)
    return np.array(projected[0], dtype=float), normalize(normal)


def integrate_tight_spiral(
    mesh: trimesh.Trimesh,
    start_point: np.ndarray,
    axis_direction: np.ndarray,
    axis_origin: np.ndarray,
    params: TightSpiralODEParams,
    num_points: int,
) -> np.ndarray:
    params.validate()
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    axis = normalize(np.array(axis_direction, dtype=float))
    origin = np.array(axis_origin, dtype=float)
    current, normal = _nearest_on_surface(mesh, np.array(start_point, dtype=float))

    points: list[np.ndarray] = [current]
    previous_direction: np.ndarray | None = None

    for _ in range(num_points - 1):
        tau, eta = make_tangent_frame(normal, axis)

        circumference = estimate_circumference(current, origin, axis)
        a = params.tangential_speed_world
        b = a * params.spacing_world / max(circumference, params.spacing_world)

        history = np.array(points[: max(0, len(points) - params.repulsion_lag_points)], dtype=float)
        grad_r = repulsive_potential_gradient(
            point=current,
            history=history,
            range_world=params.repulsion_range_world,
        )

        velocity = a * tau + b * eta - params.repulsion_strength * grad_r
        velocity = project_to_tangent_plane(velocity, normal)
        if float(np.linalg.norm(velocity)) < 1e-12:
            velocity = eta

        direction = clamp_direction_by_curvature(
            previous_direction=previous_direction,
            candidate_direction=velocity,
            step_size_world=params.step_size_world,
            min_radius_world=params.min_curvature_radius_world,
        )

        candidate = current + params.step_size_world * direction
        next_point, next_normal = _nearest_on_surface(mesh, candidate)

        step = next_point - current
        if float(np.linalg.norm(step)) < 1e-12:
            break

        previous_direction = normalize(step)
        current = next_point
        normal = next_normal
        points.append(current)

    return np.array(points, dtype=float)
