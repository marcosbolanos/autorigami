from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree
import trimesh


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    min_nm: float
    mean_nm: float
    q25_nm: float
    q75_nm: float
    max_nm: float


@dataclass(frozen=True)
class AxisCoverageSummary:
    polyline_min_nm: float
    polyline_max_nm: float
    mesh_min_nm: float
    mesh_max_nm: float
    span_nm: float
    mesh_span_nm: float
    span_ratio: float
    start_ratio: float
    end_ratio: float


@dataclass(frozen=True)
class PolylineMetrics:
    length_nm: float
    length_world: float
    nearest_nonlocal_separation: DistributionSummary
    axis_coverage: AxisCoverageSummary


def _summarize_distribution_nm(values_world: np.ndarray, world_to_nm: float) -> DistributionSummary:
    if values_world.ndim != 1 or values_world.size == 0:
        raise ValueError("values_world must be a non-empty 1D array")

    values_nm = values_world * world_to_nm
    return DistributionSummary(
        count=int(values_nm.size),
        min_nm=float(np.min(values_nm)),
        mean_nm=float(np.mean(values_nm)),
        q25_nm=float(np.quantile(values_nm, 0.25)),
        q75_nm=float(np.quantile(values_nm, 0.75)),
        max_nm=float(np.max(values_nm)),
    )


def _nearest_nonlocal_distances_world(
    points: np.ndarray,
    separation_nm: float,
    world_to_nm: float,
    neighbor_exclusion: int,
) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if pts.shape[0] < 3:
        raise ValueError("need at least 3 points")
    if world_to_nm <= 0:
        raise ValueError("world_to_nm must be > 0")
    if separation_nm <= 0:
        raise ValueError("separation_nm must be > 0")
    if neighbor_exclusion < 0:
        raise ValueError("neighbor_exclusion must be >= 0")

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate(([0.0], np.cumsum(seg)))
    min_arclen_gap_world = separation_nm / world_to_nm

    tree = KDTree(pts)
    k = min(512, pts.shape[0])
    dists, idxs = tree.query(pts, k=k)
    dist_rows = np.asarray(dists, dtype=float)
    idx_rows = np.asarray(idxs, dtype=np.int64)

    nearest: list[float] = []
    for i in range(pts.shape[0]):
        selected_distance: float | None = None
        for dist, j in zip(dist_rows[i], idx_rows[i]):
            idx = int(j)
            if idx == i:
                continue
            if abs(idx - i) <= neighbor_exclusion:
                continue
            if abs(float(arclen[idx] - arclen[i])) < min_arclen_gap_world:
                continue
            selected_distance = float(dist)
            break
        if selected_distance is None:
            fallback_dists, fallback_idxs = tree.query(pts[i], k=pts.shape[0])
            fallback_dist_row = np.asarray(fallback_dists, dtype=float)
            fallback_idx_row = np.asarray(fallback_idxs, dtype=np.int64)
            for dist, j in zip(fallback_dist_row, fallback_idx_row):
                idx = int(j)
                if idx == i:
                    continue
                if abs(idx - i) <= neighbor_exclusion:
                    continue
                if abs(float(arclen[idx] - arclen[i])) < min_arclen_gap_world:
                    continue
                selected_distance = float(dist)
                break
        if selected_distance is not None:
            nearest.append(selected_distance)

    if not nearest:
        raise ValueError("could not find any valid nonlocal neighbors")
    return np.asarray(nearest, dtype=float)


def _compute_axis_coverage(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    axis_direction: np.ndarray,
    axis_origin: np.ndarray,
    world_to_nm: float,
) -> AxisCoverageSummary:
    pts = np.asarray(points, dtype=float)
    axis = np.asarray(axis_direction, dtype=float)
    origin = np.asarray(axis_origin, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if axis.shape != (3,):
        raise ValueError("axis_direction must have shape (3,)")
    if origin.shape != (3,):
        raise ValueError("axis_origin must have shape (3,)")
    if world_to_nm <= 0:
        raise ValueError("world_to_nm must be > 0")

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0:
        raise ValueError("axis_direction must be non-zero")
    axis_unit = axis / axis_norm

    mesh_axis_positions = (np.asarray(mesh.vertices, dtype=float) - origin) @ axis_unit
    polyline_axis_positions = (pts - origin) @ axis_unit

    mesh_min_world = float(np.min(mesh_axis_positions))
    mesh_max_world = float(np.max(mesh_axis_positions))
    mesh_span_world = mesh_max_world - mesh_min_world
    if mesh_span_world <= 0:
        raise ValueError("mesh axis span must be > 0")

    polyline_min_world = float(np.min(polyline_axis_positions))
    polyline_max_world = float(np.max(polyline_axis_positions))
    polyline_span_world = polyline_max_world - polyline_min_world

    return AxisCoverageSummary(
        polyline_min_nm=polyline_min_world * world_to_nm,
        polyline_max_nm=polyline_max_world * world_to_nm,
        mesh_min_nm=mesh_min_world * world_to_nm,
        mesh_max_nm=mesh_max_world * world_to_nm,
        span_nm=polyline_span_world * world_to_nm,
        mesh_span_nm=mesh_span_world * world_to_nm,
        span_ratio=float(polyline_span_world / mesh_span_world),
        start_ratio=float((polyline_min_world - mesh_min_world) / mesh_span_world),
        end_ratio=float((polyline_max_world - mesh_min_world) / mesh_span_world),
    )


def compute_polyline_metrics(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    axis_direction: np.ndarray,
    axis_origin: np.ndarray,
    world_to_nm: float,
    separation_nm: float,
    neighbor_exclusion: int = 8,
) -> PolylineMetrics:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if pts.shape[0] < 3:
        raise ValueError("need at least 3 points")

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    nearest_nonlocal = _nearest_nonlocal_distances_world(
        points=pts,
        separation_nm=separation_nm,
        world_to_nm=world_to_nm,
        neighbor_exclusion=neighbor_exclusion,
    )

    return PolylineMetrics(
        length_nm=float(np.sum(seg) * world_to_nm),
        length_world=float(np.sum(seg)),
        nearest_nonlocal_separation=_summarize_distribution_nm(nearest_nonlocal, world_to_nm),
        axis_coverage=_compute_axis_coverage(
            points=pts,
            mesh=mesh,
            axis_direction=axis_direction,
            axis_origin=axis_origin,
            world_to_nm=world_to_nm,
        ),
    )
