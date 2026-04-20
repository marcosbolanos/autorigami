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


def _nearest_on_surface(
    mesh: trimesh.Trimesh, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    projected, _, face_idx = mesh.nearest.on_surface(point[None, :])
    idx = int(face_idx[0])
    normal = np.array(mesh.face_normals[idx], dtype=float)
    return np.array(projected[0], dtype=float), normalize(normal)


def _advance_with_progress_guard(
    mesh: trimesh.Trimesh,
    current: np.ndarray,
    base_direction: np.ndarray,
    eta: np.ndarray,
    axis: np.ndarray,
    params: TightSpiralODEParams,
    min_axis_step_world: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    trial_blends = np.linspace(0.0, 1.0, 9, dtype=float)
    for blend in trial_blends:
        trial_direction = normalize((1.0 - blend) * base_direction + blend * eta)
        candidate = current + params.step_size_world * trial_direction
        next_point, next_normal = _nearest_on_surface(mesh, candidate)
        step = next_point - current
        step_norm = float(np.linalg.norm(step))
        if step_norm < 1e-12:
            continue
        axis_step_world = float(np.dot(step, axis))
        if axis_step_world >= min_axis_step_world:
            return next_point, next_normal, step
    return None


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
    vertices = np.asarray(mesh.vertices, dtype=float)
    axis_positions = (vertices - origin) @ axis
    axis_min_world = float(np.min(axis_positions))
    axis_max_world = float(np.max(axis_positions))
    top_limit_world = axis_max_world - params.top_clearance_world
    current, normal = _nearest_on_surface(mesh, np.array(start_point, dtype=float))

    points: list[np.ndarray] = [current]
    previous_direction: np.ndarray | None = None

    for _ in range(num_points - 1):
        current_axis_world = float(np.dot(current - origin, axis))
        if current_axis_world >= top_limit_world:
            break

        tau, eta = make_tangent_frame(normal, axis)

        circumference = estimate_circumference(current, origin, axis)
        a = params.tangential_speed_world
        b = a * params.spacing_world / max(circumference, params.spacing_world)

        history = np.array(
            points[: max(0, len(points) - params.repulsion_lag_points)], dtype=float
        )
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

        ideal_axis_step_world = (
            params.step_size_world
            * params.spacing_world
            / max(
                circumference,
                params.spacing_world,
            )
        )
        min_axis_step_world = params.min_progress_fraction * ideal_axis_step_world
        if current_axis_world <= axis_min_world + params.step_size_world:
            min_axis_step_world = 0.0
        advance = _advance_with_progress_guard(
            mesh=mesh,
            current=current,
            base_direction=direction,
            eta=eta,
            axis=axis,
            params=params,
            min_axis_step_world=min_axis_step_world,
        )
        if advance is None:
            break
        next_point, next_normal, step = advance

        previous_direction = normalize(step)
        current = next_point
        normal = next_normal
        points.append(current)

    return np.array(points, dtype=float)
