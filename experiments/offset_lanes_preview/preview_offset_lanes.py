from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import svgwrite  # type: ignore[reportMissingTypeStubs]
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points

from src.curvepack.svg_io import (
    load_single_path_polygon,
    load_svg_canvas,
    load_svg_scale_nm,
)


@dataclass(frozen=True)
class LaneParams:
    step_nm: float
    t0_nm: float
    rmin_nm: float
    min_strand_len_nm: float
    seam_axis: str
    simplify_nm: float
    buffer_resolution: int
    max_lanes: int


@dataclass(frozen=True)
class Strand:
    points: np.ndarray  # (N,2) float32
    lane_index: int

    def length(self) -> float:
        if self.points.shape[0] < 2:
            return 0.0
        ds = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)
        return float(np.sum(ds))


def _resolve_repo_path(rel_path: str) -> Path:
    # experiments/.. sits at repo root.
    return Path(__file__).resolve().parents[2] / rel_path


def _arclength_resample_closed(pts: np.ndarray, step: float) -> np.ndarray:
    """Resample a closed polyline (ring) at approximately uniform arclength step."""
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N,2)")
    if pts.shape[0] < 4:
        return pts.astype(np.float32)

    # Ensure closed (last == first)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total <= 1e-9:
        return pts[:1].astype(np.float32)

    n = max(3, int(math.floor(total / step)))
    s_targets = np.linspace(0.0, total, n, endpoint=False)

    cum = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_len)])
    out = np.zeros((len(s_targets), 2), dtype=np.float64)

    j = 0
    for i, s in enumerate(s_targets):
        while j + 1 < len(cum) and cum[j + 1] < s:
            j += 1
        ds = s - cum[j]
        L = float(seg_len[j])
        if L <= 1e-12:
            out[i] = pts[j]
        else:
            t = ds / L
            out[i] = (1.0 - t) * pts[j] + t * pts[j + 1]

    return out.astype(np.float32)


def _discrete_curvature(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Approximate curvature kappa at vertices for a polyline.

    Returns kappa for indices 1..N-2 (same length N-2).
    kappa ~= turning_angle / local_arclength.
    """

    u = P[1:-1] - P[0:-2]
    v = P[2:] - P[1:-1]
    nu = np.linalg.norm(u, axis=1) + eps
    nv = np.linalg.norm(v, axis=1) + eps
    cos_th = np.sum(u * v, axis=1) / (nu * nv)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    theta = np.arccos(cos_th)
    ds = 0.5 * (nu + nv)
    return theta / (ds + eps)


def _point_segment_dist2(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-12
) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab)) + eps
    t = float(np.dot(p - a, ab)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    q = a + t * ab
    d = p - q
    return float(np.dot(d, d))


def _seg_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12
) -> bool:
    # Robust-ish 2D segment intersection (with epsilon).
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    dx, dy = float(d[0]), float(d[1])

    def orient(
        px: float, py: float, qx: float, qy: float, rx: float, ry: float
    ) -> float:
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True

    def on_seg(
        px: float, py: float, qx: float, qy: float, rx: float, ry: float
    ) -> bool:
        return (
            min(px, rx) - eps <= qx <= max(px, rx) + eps
            and min(py, ry) - eps <= qy <= max(py, ry) + eps
        )

    if abs(o1) <= eps and on_seg(ax, ay, cx, cy, bx, by):
        return True
    if abs(o2) <= eps and on_seg(ax, ay, dx, dy, bx, by):
        return True
    if abs(o3) <= eps and on_seg(cx, cy, ax, ay, dx, dy):
        return True
    if abs(o4) <= eps and on_seg(cx, cy, bx, by, dx, dy):
        return True
    return False


def _segseg_dist2(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    # In 2D, if segments don't intersect, the closest point involves an endpoint.
    if _seg_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_dist2(a, c, d),
        _point_segment_dist2(b, c, d),
        _point_segment_dist2(c, a, b),
        _point_segment_dist2(d, a, b),
    )


def _point_segment_closest_point(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, float, float]:
    ab = b - a
    denom = float(np.dot(ab, ab)) + eps
    t = float(np.dot(p - a, ab)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    q = a + t * ab
    d = p - q
    return q, t, float(np.dot(d, d))


def _segseg_closest_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return (p_on_ab, q_on_cd, dist, s, t).

    s is the interpolation parameter on ab: p = a + s*(b-a)
    t is the interpolation parameter on cd: q = c + t*(d-c)
    """
    if _seg_intersect(a, b, c, d):
        # Degenerate case: pick midpoints and separate along a stable normal.
        p = 0.5 * (a + b)
        q = 0.5 * (c + d)
        return p, q, 0.0, 0.5, 0.5

    qa, ta, da2 = _point_segment_closest_point(a, c, d)
    qb, tb, db2 = _point_segment_closest_point(b, c, d)
    qc, sc, dc2 = _point_segment_closest_point(c, a, b)
    qd, sd, dd2 = _point_segment_closest_point(d, a, b)

    # Track parameters (s on ab, t on cd)
    candidates = [
        (a, qa, da2, 0.0, ta),
        (b, qb, db2, 1.0, tb),
        (qc, c, dc2, sc, 0.0),
        (qd, d, dd2, sd, 1.0),
    ]
    p_best, q_best, d2, s_best, t_best = min(candidates, key=lambda x: x[2])
    return p_best, q_best, float(math.sqrt(max(d2, 0.0))), float(s_best), float(t_best)


def _project_inside_polygon(points: np.ndarray, poly: Polygon) -> None:
    # Hard projection: if a point is outside, snap to nearest point on polygon.
    # This is a last-resort guard to keep things inside during relaxation.
    for i in range(points.shape[0]):
        x, y = float(points[i, 0]), float(points[i, 1])
        pt = Point(x, y)
        poly_any = cast(Any, poly)
        if bool(poly_any.covers(pt)):
            continue
        p = nearest_points(poly_any, pt)[0]
        points[i, 0] = float(p.x)
        points[i, 1] = float(p.y)


def _project_edge_lengths(
    points: np.ndarray, step: float, stiffness: float = 1.0
) -> None:
    if points.shape[0] < 2:
        return
    for i in range(points.shape[0] - 1):
        a = points[i]
        b = points[i + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= 1e-12:
            continue
        corr = (L - step) / L
        delta = stiffness * 0.5 * corr * d
        points[i] += delta
        points[i + 1] -= delta


def _project_turning_angle_max(
    points: np.ndarray, theta_max: float, stiffness: float = 1.0
) -> None:
    if points.shape[0] < 3:
        return
    for i in range(1, points.shape[0] - 1):
        a = points[i - 1]
        p = points[i]
        b = points[i + 1]
        u = p - a
        v = b - p
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu <= 1e-9 or nv <= 1e-9:
            continue
        cos_th = float(np.dot(u, v) / (nu * nv))
        cos_th = float(np.clip(cos_th, -1.0, 1.0))
        th = float(math.acos(cos_th))
        if th <= theta_max:
            continue
        # Move p towards the midpoint of neighbors to reduce turning.
        target = 0.5 * (a + b)
        alpha = stiffness * (th - theta_max) / max(th, 1e-9)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        points[i] = (1.0 - alpha) * p + alpha * target


def relax_strands(
    strands: list[Strand],
    *,
    poly: Polygon,
    step: float,
    rmin: float,
    min_sep: float,
    eps: float,
    iters: int,
    exclude_adj_within_strand: int,
) -> list[Strand]:
    """Projected relaxation to satisfy curvature + separation with zero tolerance.

    This is a PBD-style solver: iteratively project constraints.
    """

    if iters <= 0:
        return strands
    if step <= 0 or rmin <= 0 or min_sep <= 0:
        raise ValueError("step, rmin, and min_sep must be positive")

    theta_max = step / rmin
    min_sep_eff = min_sep - eps
    min_sep2 = min_sep_eff * min_sep_eff
    cell = float(min_sep)

    # Work on mutable copies
    pts_list = [s.points.astype(np.float64).copy() for s in strands]

    for it in range(iters):
        # 1) Enforce separation between segments (and within strand for far-apart segments).
        # Build buckets of segments by midpoint.
        mins: np.ndarray | None = None
        buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
        mids: list[np.ndarray] = []
        seg_index: list[tuple[int, int]] = []

        for sid, pts in enumerate(pts_list):
            if pts.shape[0] < 2:
                continue
            A = pts[:-1]
            B = pts[1:]
            mid = 0.5 * (A + B)
            if mins is None:
                mins = mid.min(axis=0)
            else:
                mins = np.minimum(mins, mid.min(axis=0))
            for k in range(mid.shape[0]):
                mids.append(mid[k])
                seg_index.append((sid, k))

        if mins is None:
            continue

        mins = mins.astype(np.float64)

        for idx, m in enumerate(mids):
            gx = int(math.floor(float((m[0] - mins[0]) / cell)))
            gy = int(math.floor(float((m[1] - mins[1]) / cell)))
            buckets.setdefault((gx, gy), []).append(seg_index[idx])

        # Process pairs once per iteration.
        processed: set[tuple[int, int, int, int]] = set()
        for (gx, gy), segs in buckets.items():
            # gather from neighboring buckets
            cand: list[tuple[int, int]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(buckets.get((gx + dx, gy + dy), []))
            for sid, k in segs:
                pts = pts_list[sid]
                a = pts[k]
                b = pts[k + 1]
                for sjd, l in cand:
                    if sjd < sid or (sjd == sid and l <= k):
                        continue
                    if sid == sjd and abs(k - l) <= exclude_adj_within_strand:
                        continue
                    key = (sid, k, sjd, l)
                    if key in processed:
                        continue
                    processed.add(key)
                    pts2 = pts_list[sjd]
                    c = pts2[l]
                    d = pts2[l + 1]

                    # Quick midpoint reject
                    m1 = 0.5 * (a + b)
                    m2 = 0.5 * (c + d)
                    if float(np.dot(m1 - m2, m1 - m2)) > (4.0 * min_sep2):
                        continue

                    p_cl, q_cl, dist, s_ab, t_cd = _segseg_closest_points(a, b, c, d)
                    if dist >= min_sep_eff:
                        continue
                    # Compute separation direction.
                    n = p_cl - q_cl
                    n_norm = float(np.linalg.norm(n))
                    if n_norm <= 1e-12:
                        # Fall back to midpoint direction.
                        n = m1 - m2
                        n_norm = float(np.linalg.norm(n))
                    if n_norm <= 1e-12:
                        # As a last resort, use a perpendicular to segment ab.
                        tvec = b - a
                        n = np.array([-tvec[1], tvec[0]], dtype=np.float64)
                        n_norm = float(np.linalg.norm(n))
                    if n_norm <= 1e-12:
                        continue
                    n = n / n_norm

                    gap = float(max(min_sep_eff - dist, 0.0))
                    dp = 0.5 * gap * n
                    dq = -0.5 * gap * n

                    # Distribute translation to endpoints by closest-point barycentric weights.
                    pts_list[sid][k] += dp * (1.0 - s_ab)
                    pts_list[sid][k + 1] += dp * s_ab
                    pts_list[sjd][l] += dq * (1.0 - t_cd)
                    pts_list[sjd][l + 1] += dq * t_cd

        # 2) Enforce max turning (curvature proxy).
        for pts in pts_list:
            _project_turning_angle_max(pts, theta_max=theta_max, stiffness=0.7)

        # 3) Light regularization towards uniform sampling and keep inside polygon.
        for pts in pts_list:
            _project_edge_lengths(pts, step=step, stiffness=0.2)
            _project_inside_polygon(pts, poly)

        if (it % 10) == 0 and iters >= 20:
            print(f"relax it={it}/{iters}")

    out: list[Strand] = []
    for s, pts in zip(strands, pts_list, strict=True):
        out.append(Strand(points=pts.astype(np.float32), lane_index=s.lane_index))
    return out


def _iter_segments(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pts.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return pts[:-1].astype(np.float32), pts[1:].astype(np.float32)


def _pack_strands_with_trimming(
    candidates: list[Strand],
    *,
    min_sep: float,
    eps: float,
    exclude_adj_within_strand: int,
    guard_segments: int,
    min_points: int,
) -> tuple[list[Strand], list[Strand]]:
    """Greedy pack with local trimming on conflicts.

    Instead of rejecting an entire candidate strand on the first collision with the
    accepted set, we remove the colliding segment(s) (plus a guard window), split the
    strand, and keep any remaining collision-free pieces.
    """

    if min_sep <= 0:
        raise ValueError("min_sep must be positive")

    cell = float(min_sep)
    min_sep2 = float((min_sep - eps) * (min_sep - eps))

    accepted: list[Strand] = []
    rejected: list[Strand] = []

    mins: np.ndarray | None = None
    buckets: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}

    def seg_key(mid: np.ndarray) -> tuple[int, int]:
        assert mins is not None
        gx = int(math.floor(float((mid[0] - mins[0]) / cell)))
        gy = int(math.floor(float((mid[1] - mins[1]) / cell)))
        return gx, gy

    def add_segments(P: np.ndarray) -> None:
        nonlocal mins
        A, B = _iter_segments(P)
        if A.shape[0] == 0:
            return
        mid = 0.5 * (A + B)
        if mins is None:
            mins = mid.min(axis=0).astype(np.float64)
        for a, b, m in zip(A, B, mid, strict=True):
            key = seg_key(m)
            buckets.setdefault(key, []).append(
                (a.astype(np.float64), b.astype(np.float64))
            )

    def self_conflict(P: np.ndarray) -> bool:
        # Same as in _pack_strands_by_separation
        A, B = _iter_segments(P)
        S = A.shape[0]
        if S <= 2:
            return False
        mid = 0.5 * (A + B)
        mins_local = mid.min(axis=0).astype(np.float64)
        local: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for idx in range(S):
            m = mid[idx]
            gx = int(math.floor(float((m[0] - mins_local[0]) / cell)))
            gy = int(math.floor(float((m[1] - mins_local[1]) / cell)))
            local.setdefault((gx, gy), []).append(
                (idx, A[idx].astype(np.float64), B[idx].astype(np.float64))
            )
        for idx in range(S):
            m = mid[idx]
            gx = int(math.floor(float((m[0] - mins_local[0]) / cell)))
            gy = int(math.floor(float((m[1] - mins_local[1]) / cell)))
            a64 = A[idx].astype(np.float64)
            b64 = B[idx].astype(np.float64)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j, c64, d64 in local.get((gx + dx, gy + dy), []):
                        if abs(j - idx) <= exclude_adj_within_strand:
                            continue
                        if _segseg_dist2(a64, b64, c64, d64) < min_sep2:
                            return True
        return False

    def conflicting_segments(P: np.ndarray) -> set[int]:
        """Return set of segment indices in P that conflict with accepted set."""
        if mins is None:
            return set()
        A, B = _iter_segments(P)
        if A.shape[0] == 0:
            return set()
        mid = 0.5 * (A + B)
        bad: set[int] = set()
        for k in range(A.shape[0]):
            bx, by = seg_key(mid[k])
            a64 = A[k].astype(np.float64)
            b64 = B[k].astype(np.float64)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for c64, d64 in buckets.get((bx + dx, by + dy), []):
                        if _segseg_dist2(a64, b64, c64, d64) < min_sep2:
                            bad.add(k)
                            break
                    if k in bad:
                        break
                if k in bad:
                    break
        return bad

    def split_by_bad_segments(P: np.ndarray, bad: set[int]) -> list[np.ndarray]:
        if P.shape[0] < 2:
            return []
        S = P.shape[0] - 1
        seg_bad = np.zeros((S,), dtype=bool)
        for k in bad:
            s0 = max(0, k - guard_segments)
            s1 = min(S, k + guard_segments + 1)
            seg_bad[s0:s1] = True
        point_good = np.ones((P.shape[0],), dtype=bool)
        for k in np.nonzero(seg_bad)[0]:
            point_good[k] = False
            point_good[k + 1] = False

        segs: list[np.ndarray] = []
        start = None
        for i, ok in enumerate(point_good):
            if ok and start is None:
                start = i
            if (not ok or i == point_good.size - 1) and start is not None:
                end = i if not ok else i + 1
                if end - start >= min_points:
                    segs.append(P[start:end].astype(np.float32))
                start = None
        return segs

    ordered = sorted(candidates, key=lambda s: (-s.length(), s.lane_index))
    for strand in ordered:
        pieces = [strand.points]
        kept_any = False
        while pieces:
            P = pieces.pop(0)
            if P.shape[0] < min_points:
                continue
            if self_conflict(P):
                continue
            bad = conflicting_segments(P)
            if not bad:
                accepted.append(
                    Strand(points=P.astype(np.float32), lane_index=strand.lane_index)
                )
                add_segments(P)
                kept_any = True
                continue
            pieces.extend(split_by_bad_segments(P, bad))

        if not kept_any:
            rejected.append(strand)

    return accepted, rejected


def _cut_closed_polyline_at_index(P: np.ndarray, cut_idx: int) -> np.ndarray:
    """Cut a closed polyline (no repeated last point required) into an open one."""
    if P.shape[0] < 3:
        return P
    cut_idx = int(cut_idx) % P.shape[0]
    return np.vstack([P[cut_idx:], P[: cut_idx + 1]]).astype(np.float32)


def _choose_seam_cut_index(P: np.ndarray, axis: str) -> int:
    if axis not in ("x", "y"):
        raise ValueError("seam_axis must be 'x' or 'y'")
    c = P.mean(axis=0)
    if axis == "x":
        seam = c[0]
        return int(np.argmin(np.abs(P[:, 0] - seam)))
    seam = c[1]
    return int(np.argmin(np.abs(P[:, 1] - seam)))


def _split_on_curvature_violations(
    P_open: np.ndarray,
    *,
    kappa_max: float,
    min_points: int,
    guard: int,
) -> list[np.ndarray]:
    """Split an open polyline into segments where curvature exceeds kappa_max.

    guard: number of points to drop on each side of a violating vertex.
    """

    if P_open.shape[0] < (min_points + 2):
        return []

    kappa = _discrete_curvature(P_open)
    bad = np.where(kappa > kappa_max)[0] + 1  # shift to vertex indices in P_open
    if bad.size == 0:
        return [P_open]

    cut = np.zeros(P_open.shape[0], dtype=bool)
    for idx in bad:
        lo = max(0, idx - guard)
        hi = min(P_open.shape[0], idx + guard + 1)
        cut[lo:hi] = True

    # Keep contiguous runs of ~cut == False
    keep = ~cut
    segs: list[np.ndarray] = []
    start = None
    for i, ok in enumerate(keep):
        if ok and start is None:
            start = i
        if (not ok or i == keep.size - 1) and start is not None:
            end = i if not ok else i + 1
            if end - start >= min_points:
                segs.append(P_open[start:end].astype(np.float32))
            start = None
    return segs


def _extract_rings(g: object) -> list[np.ndarray]:
    """Extract exterior rings from a shapely geometry as Nx2 float32 arrays."""
    if g is None:
        return []
    if isinstance(g, Polygon):
        if g.is_empty:
            return []
        coords = np.asarray(g.exterior.coords, dtype=np.float32)
        if coords.shape[0] < 3:
            return []
        return [coords]
    if isinstance(g, MultiPolygon):
        out: list[np.ndarray] = []
        for p in g.geoms:
            out.extend(_extract_rings(p))
        return out
    if isinstance(g, GeometryCollection):
        out = []
        for gg in g.geoms:
            out.extend(_extract_rings(gg))
        return out
    return []


def _default_viewbox(
    points: np.ndarray, pad: float
) -> tuple[float, float, float, float]:
    minx, miny = points.min(axis=0)
    maxx, maxy = points.max(axis=0)
    return (
        float(minx - pad),
        float(miny - pad),
        float((maxx - minx) + 2.0 * pad),
        float((maxy - miny) + 2.0 * pad),
    )


def _export_preview_svg(
    out_path: Path,
    *,
    V: np.ndarray,
    strands: list[np.ndarray],
    rejected: list[np.ndarray] | None,
    viewbox: tuple[float, float, float, float],
    canvas_size: tuple[str, str] | None,
    stroke_width: float | str,
    draw_points: bool,
    dot_r: float,
) -> None:
    dwg = (
        svgwrite.Drawing(str(out_path), profile="tiny")
        if canvas_size is None
        else svgwrite.Drawing(str(out_path), profile="tiny", size=canvas_size)
    )
    dwg.attribs["viewBox"] = f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}"

    def to_point_list(pts: np.ndarray) -> list[tuple[float, float]]:
        return [(float(p[0]), float(p[1])) for p in pts]

    dwg.add(
        dwg.polygon(
            points=to_point_list(V),
            stroke="#111",
            fill="none",
            stroke_width=stroke_width,
            opacity=0.9,
        )
    )

    if rejected is not None and len(rejected) > 0:
        gr = dwg.g(id="rejected_strands", fill="none", stroke="#999", opacity=0.35)
        for s in rejected:
            if s.shape[0] < 2:
                continue
            gr.add(
                dwg.polyline(
                    points=to_point_list(s),
                    stroke="#999",
                    fill="none",
                    stroke_width=stroke_width,
                    stroke_linecap="round",
                    stroke_linejoin="round",
                )
            )
        dwg.add(gr)

    g = dwg.g(id="offset_strands", fill="none", stroke="#d11", opacity=0.85)
    for s in strands:
        if s.shape[0] < 2:
            continue
        g.add(
            dwg.polyline(
                points=to_point_list(s),
                stroke="#d11",
                fill="none",
                stroke_width=stroke_width,
                stroke_linecap="round",
                stroke_linejoin="round",
            )
        )
    dwg.add(g)

    if draw_points:
        gp = dwg.g(id="offset_points", fill="#d11", stroke="none", opacity=0.7)
        for s in strands:
            for x, y in s:
                gp.add(dwg.circle(center=(float(x), float(y)), r=float(dot_r)))
        dwg.add(gp)

    dwg.save()


def generate_offset_lane_strands(
    V: np.ndarray,
    *,
    nm_per_unit: float,
    params: LaneParams,
) -> list[Strand]:
    poly = Polygon(V)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return []

    step_units = params.step_nm / nm_per_unit
    if params.simplify_nm > 0:
        simplify_units = params.simplify_nm / nm_per_unit
        # preserve_topology avoids introducing self-intersections.
        poly = poly.simplify(simplify_units, preserve_topology=True)
        if poly.is_empty:
            return []

    t0_units = params.t0_nm / nm_per_unit
    rmin_units = params.rmin_nm / nm_per_unit
    min_len_units = params.min_strand_len_nm / nm_per_unit

    kappa_max = 1.0 / rmin_units
    guard = 2  # drop a couple of points around violations
    min_points = max(4, int(math.ceil(min_len_units / step_units)))

    strands: list[Strand] = []
    k = 0
    while k < params.max_lanes:
        t = t0_units + k * step_units
        inner = poly.buffer(-t, resolution=int(params.buffer_resolution))
        if getattr(inner, "is_empty", False):
            break
        rings = _extract_rings(inner)
        if not rings:
            break

        for ring in rings:
            # Shapely rings repeat last point == first. Remove last for resampling.
            if ring.shape[0] >= 2 and np.linalg.norm(ring[0] - ring[-1]) < 1e-6:
                ring = ring[:-1]
            if ring.shape[0] < 3:
                continue

            ring_rs = _arclength_resample_closed(ring, step=step_units)
            if ring_rs.shape[0] < 8:
                continue

            cut_idx = _choose_seam_cut_index(ring_rs, axis=params.seam_axis)
            open_lane = _cut_closed_polyline_at_index(ring_rs, cut_idx)

            segs = _split_on_curvature_violations(
                open_lane,
                kappa_max=kappa_max,
                min_points=min_points,
                guard=guard,
            )
            for seg in segs:
                if seg.shape[0] < min_points:
                    continue
                # Drop segments that are too short in arclength.
                ds = np.linalg.norm(seg[1:] - seg[:-1], axis=1)
                if float(np.sum(ds)) < min_len_units:
                    continue
                strands.append(Strand(points=seg.astype(np.float32), lane_index=k))

        if (k % 5) == 0:
            print(f"offset_lane k={k} t_units={t:.4g} components={len(rings)}")
        k += 1

    return strands


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview boundary-parallel offset lane strands (simple, hairpin-averse)."
    )
    ap.add_argument(
        "--input",
        type=str,
        default=str(_resolve_repo_path("data/raw/blob1.svg")),
        help="Input SVG path",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(
            _resolve_repo_path("data/processed/blob1_offset_lanes_preview.svg")
        ),
        help="Output preview SVG path",
    )
    ap.add_argument(
        "--step_nm",
        type=float,
        default=2.6,
        help="Lane spacing and resample step (nm)",
    )
    ap.add_argument(
        "--t0_nm",
        type=float,
        default=1.3,
        help="First inset from boundary (nm). Default is half step.",
    )
    ap.add_argument(
        "--rmin_nm",
        type=float,
        default=6.0,
        help="Minimum allowed bend radius (nm)",
    )
    ap.add_argument(
        "--min_strand_len_nm",
        type=float,
        default=20.0,
        help="Drop strands shorter than this length (nm)",
    )
    ap.add_argument(
        "--simplify_nm",
        type=float,
        default=0.5,
        help="Simplify polygon before buffering (nm). 0 disables.",
    )
    ap.add_argument(
        "--buffer_resolution",
        type=int,
        default=8,
        help="Shapely buffer resolution (lower is faster/rougher)",
    )
    ap.add_argument(
        "--max_lanes",
        type=int,
        default=200,
        help="Maximum number of lane offsets to attempt",
    )
    ap.add_argument(
        "--scale_nm_per_cm",
        type=float,
        default=1.0,
        help="Real nm per SVG cm (default: 1cm -> 1nm)",
    )
    ap.add_argument(
        "--flat_tol_nm",
        type=float,
        default=2.0,
        help="Flatten tolerance for SVG path (nm)",
    )
    ap.add_argument(
        "--seam_axis",
        type=str,
        default="x",
        choices=["x", "y"],
        help="Axis for choosing a seam cut on each loop (makes strands open)",
    )
    ap.add_argument(
        "--stroke_width",
        type=str,
        default="0.1",
        help="Stroke width for preview SVG",
    )
    ap.add_argument(
        "--min_sep_nm",
        type=float,
        default=2.6,
        help="Hard minimum separation between any two strand segments (nm)",
    )
    ap.add_argument(
        "--sep_eps_nm",
        type=float,
        default=1e-3,
        help="Epsilon slack for separation comparisons (nm)",
    )
    ap.add_argument(
        "--exclude_adj_within_strand",
        type=int,
        default=2,
        help="For self-separation check, ignore segment pairs with |i-j| <= this",
    )
    ap.add_argument(
        "--show_rejected",
        action="store_true",
        help="Render rejected strands (gray) for debugging",
    )
    ap.add_argument(
        "--draw_points",
        action="store_true",
        help="Also draw points as circles (can be heavy)",
    )
    ap.add_argument(
        "--dot_r_nm",
        type=float,
        default=0.2,
        help="Point radius if --draw_points is set (nm)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input SVG not found: {in_path}")

    viewbox, canvas_size = load_svg_canvas(str(in_path))
    nm_per_unit = load_svg_scale_nm(str(in_path), nm_per_cm=float(args.scale_nm_per_cm))
    if nm_per_unit is None or nm_per_unit <= 0:
        nm_per_unit = 1.0
        print("WARNING: svg has no physical units; assuming 1 nm per SVG unit")
    else:
        print(
            "svg_scale "
            f"nm_per_unit={nm_per_unit:.6g} nm_per_cm={float(args.scale_nm_per_cm):.6g}"
        )

    flat_tol_units = float(args.flat_tol_nm) / float(nm_per_unit)
    V = load_single_path_polygon(str(in_path), flat_tol=flat_tol_units)

    params = LaneParams(
        step_nm=float(args.step_nm),
        t0_nm=float(args.t0_nm),
        rmin_nm=float(args.rmin_nm),
        min_strand_len_nm=float(args.min_strand_len_nm),
        seam_axis=str(args.seam_axis),
        simplify_nm=float(args.simplify_nm),
        buffer_resolution=int(args.buffer_resolution),
        max_lanes=int(args.max_lanes),
    )

    strands = generate_offset_lane_strands(
        V, nm_per_unit=float(nm_per_unit), params=params
    )
    print(f"lanes: candidates={len(strands)}")

    # Relaxation stage: try to fix near-violations instead of dropping strands.
    poly = Polygon(V)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise ValueError("Polygon is empty after fixing")

    step_units = float(args.step_nm) / float(nm_per_unit)
    rmin_units = float(args.rmin_nm) / float(nm_per_unit)
    min_sep_units = float(args.min_sep_nm) / float(nm_per_unit)
    sep_eps_units = float(args.sep_eps_nm) / float(nm_per_unit)

    relax_iters = 200
    print(f"relax: iters={relax_iters}")
    strands_relaxed = relax_strands(
        strands,
        poly=poly,
        step=step_units,
        rmin=rmin_units,
        min_sep=min_sep_units,
        eps=sep_eps_units,
        iters=relax_iters,
        exclude_adj_within_strand=int(args.exclude_adj_within_strand),
    )

    min_len_units = float(args.min_strand_len_nm) / float(nm_per_unit)
    min_points = max(4, int(math.ceil(min_len_units / max(step_units, 1e-9))))

    accepted, rejected = _pack_strands_with_trimming(
        strands_relaxed,
        min_sep=min_sep_units,
        eps=sep_eps_units,
        exclude_adj_within_strand=int(args.exclude_adj_within_strand),
        guard_segments=1,
        min_points=min_points,
    )
    print(
        "pack: "
        f"accepted={len(accepted)} rejected={len(rejected)} "
        f"min_sep_nm={float(args.min_sep_nm):.6g}"
    )

    if viewbox is None:
        pad_units = 2.0 * (params.step_nm / nm_per_unit)
        viewbox = _default_viewbox(V, pad=pad_units)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dot_r_units = float(args.dot_r_nm) / float(nm_per_unit)

    _export_preview_svg(
        out_path,
        V=V,
        strands=[s.points for s in accepted],
        rejected=([s.points for s in rejected] if bool(args.show_rejected) else None),
        viewbox=viewbox,
        canvas_size=canvas_size,
        stroke_width=args.stroke_width,
        draw_points=bool(args.draw_points),
        dot_r=dot_r_units,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
