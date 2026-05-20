from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from autorigami._native import validate_polyline_nonlocal_distance
from autorigami.parametrization import Polyline


@dataclass(frozen=True)
class GlobalSpiralCandidate:
    name: str
    points: np.ndarray
    turns: float
    phase: float
    axial_margin: float
    length_world: float
    nonlocal_violations: int
    minimum_distance_world: float
    axis_direction: np.ndarray
    axis_origin: np.ndarray


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def _axis_frame(axis_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = _normalize(np.asarray(axis_direction, dtype=np.float64))
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _normalize(helper - axis * float(np.dot(helper, axis)))
    v = _normalize(np.cross(axis, u))
    return axis, u, v


def _polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))


@dataclass(frozen=True)
class EllipsoidModel:
    center: np.ndarray
    basis: np.ndarray
    radii: np.ndarray


def _fit_ellipsoid_model(mesh: trimesh.Trimesh) -> EllipsoidModel:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    centered = vertices - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0.0:
        basis[:, 2] *= -1.0

    local = centered @ basis
    radii = np.max(np.abs(local), axis=0)
    if np.any(radii <= 1e-12):
        raise ValueError("Mesh must have positive extent on all fitted ellipsoid axes")
    return EllipsoidModel(center=center, basis=basis, radii=radii)


def _candidate_axis_indices(model: EllipsoidModel) -> list[int]:
    order = list(np.argsort(model.radii)[::-1])
    return order


def _curve_on_implicit_ellipsoid(
    radii: np.ndarray,
    axis_origin: np.ndarray,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    turns: float,
    phase: float,
    axial_margin: float,
    samples: int,
) -> np.ndarray:
    axis, u, v = frame
    half_length = float(radii[0])
    t = np.linspace(-1.0 + axial_margin, 1.0 - axial_margin, samples)
    theta = phase + 2.0 * np.pi * turns * (t + 1.0 - axial_margin) / (2.0 - 2.0 * axial_margin)
    axial = half_length * t
    cross_scale = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    y = float(radii[1]) * cross_scale * np.cos(theta)
    z = float(radii[2]) * cross_scale * np.sin(theta)
    return axis_origin + axial[:, None] * axis + y[:, None] * u + z[:, None] * v


def _frame_from_basis_columns(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _normalize(basis[:, 0]),
        _normalize(basis[:, 1]),
        _normalize(basis[:, 2]),
    )


def _candidate_is_valid(
    points: np.ndarray,
    spacing_world: float,
    nonlocal_window_world: float,
) -> tuple[bool, int, float]:
    result = validate_polyline_nonlocal_distance(
        points=points,
        minimum_separation=spacing_world,
        nonlocal_window=nonlocal_window_world,
        stop_on_first_violation=False,
    )
    violations = int(result["violation_count"])
    minimum_distance = float(result["minimum_checked_distance"])
    return violations == 0, violations, minimum_distance


def generate_global_spiral_candidates(
    mesh: trimesh.Trimesh,
    *,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    spacing_world: float,
    nonlocal_window_world: float,
    samples: int,
) -> list[GlobalSpiralCandidate]:
    model = _fit_ellipsoid_model(mesh)
    search_samples = min(samples, 1600)
    output_samples = min(samples, 4000)

    candidate_specs: list[tuple[float, float, float, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    requested_axis = _normalize(np.asarray(axis_direction, dtype=np.float64))
    requested_origin = np.asarray(axis_origin, dtype=np.float64)
    axis_options: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for axis_index in _candidate_axis_indices(model):
        axis = model.basis[:, axis_index]
        remaining = [idx for idx in range(3) if idx != axis_index]
        frame_basis = np.column_stack((axis, model.basis[:, remaining[0]], model.basis[:, remaining[1]]))
        frame_radii = np.array(
            [model.radii[axis_index], model.radii[remaining[0]], model.radii[remaining[1]]],
            dtype=np.float64,
        )
        frame = _frame_from_basis_columns(frame_basis)
        axis_options.append((f"pca_{axis_index}_pos", axis, model.center, frame_radii, frame))
        neg_frame = (-frame[0], frame[1], -frame[2])
        axis_options.append((f"pca_{axis_index}_neg", -axis, model.center, frame_radii, neg_frame))

    axis_options.append(("requested", requested_axis, requested_origin, model.radii, _axis_frame(requested_axis)))

    for _, axis, origin, radii, frame in axis_options:
        for margin in [0.04, 0.07, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.36]:
            for phase in np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False):
                low = 0.5
                high = 180.0
                best_turns: float | None = None
                for _ in range(18):
                    mid = 0.5 * (low + high)
                    points = _curve_on_implicit_ellipsoid(
                        radii,
                        origin,
                        frame,
                        turns=mid,
                        phase=float(phase),
                        axial_margin=margin,
                        samples=search_samples,
                    )
                    valid, _, _ = _candidate_is_valid(points, spacing_world, nonlocal_window_world)
                    if valid:
                        best_turns = mid
                        low = mid
                    else:
                        high = mid
                if best_turns is not None:
                    candidate_specs.append((best_turns, float(phase), margin, axis, origin, radii, frame))

    out: list[GlobalSpiralCandidate] = []
    for index, (turns, phase, margin, axis, origin, radii, frame) in enumerate(candidate_specs):
        points = _curve_on_implicit_ellipsoid(
            radii,
            origin,
            frame,
            turns=turns,
            phase=phase,
            axial_margin=margin,
            samples=output_samples,
        )
        for _ in range(12):
            valid, _, _ = _candidate_is_valid(points, spacing_world, nonlocal_window_world)
            if valid:
                break
            turns *= 0.98
            points = _curve_on_implicit_ellipsoid(
                radii,
                origin,
                frame,
                turns=turns,
                phase=phase,
                axial_margin=margin,
                samples=output_samples,
            )
        _, violations, minimum_distance = _candidate_is_valid(
            points,
            spacing_world,
            nonlocal_window_world,
        )
        out.append(
            GlobalSpiralCandidate(
                name=f"global_{index:03d}",
                points=points,
                turns=turns,
                phase=phase,
                axial_margin=margin,
                length_world=_polyline_length(points),
                nonlocal_violations=violations,
                minimum_distance_world=minimum_distance,
                axis_direction=axis,
                axis_origin=origin,
            )
        )
    return out


def to_polyline(candidate: GlobalSpiralCandidate) -> Polyline:
    return Polyline(points=np.asarray(candidate.points, dtype=np.float64))
