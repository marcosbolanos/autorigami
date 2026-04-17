from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree


@dataclass(frozen=True)
class ConstraintReport:
    compliant_count: int
    total_count: int

    @property
    def ratio(self) -> float:
        if self.total_count <= 0:
            return 0.0
        return self.compliant_count / self.total_count


@dataclass(frozen=True)
class ValidationReport:
    separation: ConstraintReport
    curvature: ConstraintReport


def _curvature_radius(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    radius = np.full(n, np.inf, dtype=float)
    if n < 3:
        return radius

    p0 = points[:-2]
    p1 = points[1:-1]
    p2 = points[2:]

    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)

    cross = np.linalg.norm(np.cross(p1 - p0, p2 - p1), axis=1)
    kappa = np.zeros_like(a)
    denom = a * b * c
    valid = denom > 1e-12
    kappa[valid] = 2.0 * cross[valid] / denom[valid]

    with np.errstate(divide="ignore"):
        local_radius = np.where(kappa > 1e-12, 1.0 / kappa, np.inf)
    radius[1:-1] = local_radius
    return radius


def validate_polyline_constraints(
    points: np.ndarray,
    world_to_nm: float,
    separation_nm: float,
    min_curvature_radius_nm: float,
    neighbor_exclusion: int = 8,
) -> ValidationReport:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if pts.shape[0] < 3:
        raise ValueError("need at least 3 points")
    if world_to_nm <= 0:
        raise ValueError("world_to_nm must be > 0")

    sep_world = separation_nm / world_to_nm
    min_radius_world = min_curvature_radius_nm / world_to_nm
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate(([0.0], np.cumsum(seg)))

    tree = KDTree(pts)
    k = min(64, pts.shape[0])
    dists, idxs = tree.query(pts, k=k)
    dist_rows = np.asarray(dists, dtype=float)
    idx_rows = np.asarray(idxs, dtype=np.int64)

    separation_ok = np.zeros(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        ok = True
        for d, j in zip(dist_rows[i], idx_rows[i]):
            idx = int(j)
            if i == idx:
                continue
            if abs(idx - i) <= neighbor_exclusion:
                continue
            if abs(float(arclen[idx] - arclen[i])) < sep_world:
                continue
            if float(d) < sep_world:
                ok = False
                break
            break
        separation_ok[i] = ok

    radius = _curvature_radius(pts)
    curvature_ok = radius >= min_radius_world

    return ValidationReport(
        separation=ConstraintReport(
            compliant_count=int(np.count_nonzero(separation_ok)),
            total_count=int(separation_ok.size),
        ),
        curvature=ConstraintReport(
            compliant_count=int(np.count_nonzero(curvature_ok)),
            total_count=int(curvature_ok.size),
        ),
    )
