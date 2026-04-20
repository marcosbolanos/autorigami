from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < _EPS:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def project_to_tangent_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vec - float(np.dot(vec, normal)) * normal


def make_tangent_frame(
    normal: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    eta = project_to_tangent_plane(axis, normal)
    if float(np.linalg.norm(eta)) < _EPS:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(helper, normal))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        eta = project_to_tangent_plane(helper, normal)
    eta = normalize(eta)
    tau = normalize(np.cross(normal, eta))
    if float(np.dot(eta, axis)) < 0.0:
        eta = -eta
        tau = -tau
    return tau, eta


def estimate_circumference(
    point: np.ndarray, axis_origin: np.ndarray, axis: np.ndarray
) -> float:
    rel = point - axis_origin
    axis_pos = float(np.dot(rel, axis))
    radial = rel - axis_pos * axis
    radius = float(np.linalg.norm(radial))
    return max(2.0 * np.pi * radius, 1e-8)


def clamp_direction_by_curvature(
    previous_direction: np.ndarray | None,
    candidate_direction: np.ndarray,
    step_size_world: float,
    min_radius_world: float,
) -> np.ndarray:
    if previous_direction is None:
        return normalize(candidate_direction)

    prev = normalize(previous_direction)
    cand = normalize(candidate_direction)
    dot = float(np.clip(np.dot(prev, cand), -1.0, 1.0))
    angle = float(np.arccos(dot))

    max_angle = step_size_world / max(min_radius_world, 1e-8)
    if angle <= max_angle:
        return cand

    t = max_angle / angle
    blended = (1.0 - t) * prev + t * cand
    return normalize(blended)
