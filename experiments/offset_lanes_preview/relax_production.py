from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import svgwrite  # type: ignore[reportMissingTypeStubs]
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from src.curvepack.svg_io import (
    load_single_path_polygon,
    load_svg_canvas,
    load_svg_scale_nm,
)
from src.curvepack.bezier import beziers_to_svg_path_d, polyline_to_cubic_beziers


@dataclass(frozen=True)
class LaneParams:
    step_nm: float = 2.6
    t0_nm: float = 1.3
    rmin_nm: float = 6.0
    min_strand_len_nm: float = 20.0
    seam_axis: str = "x"
    simplify_nm: float = 0.5
    buffer_resolution: int = 8
    max_lanes: int = 200


@dataclass
class Strand:
    points0: np.ndarray  # (N,2) float32 original
    points: np.ndarray  # (N,2) float64 mutable during relax
    lane_index: int

    def length(self) -> float:
        if self.points.shape[0] < 2:
            return 0.0
        ds = np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)
        return float(np.sum(ds))


def _resolve_repo_path(rel_path: str) -> Path:
    return Path(__file__).resolve().parents[2] / rel_path


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
    viewbox: tuple[float, float, float, float],
    canvas_size: tuple[str, str] | None,
    stroke_width: str,
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

    g = dwg.g(id="offset_strands", fill="none", stroke="#d11", opacity=0.85)
    for s in strands:
        if s.shape[0] < 2:
            continue
        segs = polyline_to_cubic_beziers(s, handle_scale=0.8, max_handle_ratio=0.5)
        d = beziers_to_svg_path_d(segs, precision=3)
        if d:
            g.add(
                dwg.path(
                    d=d,
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


def _arclength_resample_closed(pts: np.ndarray, step: float) -> np.ndarray:
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N,2)")
    if pts.shape[0] < 4:
        return pts.astype(np.float32)

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


def _cut_closed_polyline_at_index(pts: np.ndarray, cut_idx: int) -> np.ndarray:
    if pts.shape[0] < 3:
        return pts
    cut_idx = int(cut_idx) % pts.shape[0]
    # Open polyline: rotate without repeating the first point.
    return np.vstack([pts[cut_idx:], pts[:cut_idx]]).astype(np.float32)


def _choose_seam_cut_index(pts: np.ndarray, axis: str) -> int:
    if axis not in ("x", "y"):
        raise ValueError("seam_axis must be 'x' or 'y'")
    c = pts.mean(axis=0)
    if axis == "x":
        seam = c[0]
        return int(np.argmin(np.abs(pts[:, 0] - seam)))
    seam = c[1]
    return int(np.argmin(np.abs(pts[:, 1] - seam)))


def _discrete_curvature(pts: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    u = pts[1:-1] - pts[0:-2]
    v = pts[2:] - pts[1:-1]
    nu = np.linalg.norm(u, axis=1) + eps
    nv = np.linalg.norm(v, axis=1) + eps
    cos_th = np.sum(u * v, axis=1) / (nu * nv)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    theta = np.arccos(cos_th)
    ds = 0.5 * (nu + nv)
    return theta / (ds + eps)


def _split_on_curvature_violations(
    pts_open: np.ndarray,
    *,
    kappa_max: float,
    min_points: int,
    guard: int,
) -> list[np.ndarray]:
    if pts_open.shape[0] < (min_points + 2):
        return []
    kappa = _discrete_curvature(pts_open)
    bad = np.where(kappa > kappa_max)[0] + 1
    if bad.size == 0:
        return [pts_open]

    cut = np.zeros(pts_open.shape[0], dtype=bool)
    for idx in bad:
        lo = max(0, idx - guard)
        hi = min(pts_open.shape[0], idx + guard + 1)
        cut[lo:hi] = True

    keep = ~cut
    segs: list[np.ndarray] = []
    start = None
    for i, ok in enumerate(keep):
        if ok and start is None:
            start = i
        if (not ok or i == keep.size - 1) and start is not None:
            end = i if not ok else i + 1
            if end - start >= min_points:
                segs.append(pts_open[start:end].astype(np.float32))
            start = None
    return segs


def _extract_rings(g: object) -> list[np.ndarray]:
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


def generate_offset_lane_candidates(
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
    t0_units = params.t0_nm / nm_per_unit
    rmin_units = params.rmin_nm / nm_per_unit
    min_len_units = params.min_strand_len_nm / nm_per_unit
    if params.simplify_nm > 0:
        simplify_units = params.simplify_nm / nm_per_unit
        poly = poly.simplify(simplify_units, preserve_topology=True)
        if poly.is_empty:
            return []

    kappa_max = 1.0 / rmin_units
    guard = 2
    min_points = max(4, int(math.ceil(min_len_units / step_units)))

    out: list[Strand] = []
    for k in range(params.max_lanes):
        t = t0_units + k * step_units
        inner = poly.buffer(-t, resolution=int(params.buffer_resolution))
        if getattr(inner, "is_empty", False):
            break
        rings = _extract_rings(inner)
        if not rings:
            break

        for ring in rings:
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
                ds = np.linalg.norm(seg[1:] - seg[:-1], axis=1)
                if float(np.sum(ds)) < min_len_units:
                    continue
                out.append(
                    Strand(
                        points0=seg.astype(np.float32),
                        points=seg.astype(np.float64).copy(),
                        lane_index=k,
                    )
                )

    return out


def _seg_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12
) -> bool:
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


def _point_segment_closest(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, float, float]:
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-12
    t = float(np.dot(p - a, ab)) / denom
    t = float(np.clip(t, 0.0, 1.0))
    q = a + t * ab
    d = p - q
    return q, t, float(np.dot(d, d))


def _segseg_closest(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    if _seg_intersect(a, b, c, d):
        p = 0.5 * (a + b)
        q = 0.5 * (c + d)
        return p, q, 0.0
    qa, _, da2 = _point_segment_closest(a, c, d)
    qb, _, db2 = _point_segment_closest(b, c, d)
    qc, _, dc2 = _point_segment_closest(c, a, b)
    qd, _, dd2 = _point_segment_closest(d, a, b)
    p_best, q_best, d2 = min(
        (a, qa, da2),
        (b, qb, db2),
        (qc, c, dc2),
        (qd, d, dd2),
        key=lambda t: t[2],
    )
    return p_best, q_best, float(math.sqrt(max(d2, 0.0)))


def _polyline_normals(pts: np.ndarray) -> np.ndarray:
    """Return per-vertex unit normals based on local tangents."""
    N = pts.shape[0]
    if N == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if N == 1:
        return np.array([[0.0, 1.0]], dtype=np.float64)

    t = np.zeros((N, 2), dtype=np.float64)
    t[1:-1] = pts[2:] - pts[:-2]
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn = np.maximum(nn, 1e-9)
    return n / nn


def _build_segment_index(
    strands: list[Strand], cell: float
) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int]]]]:
    mids: list[np.ndarray] = []
    idx: list[tuple[int, int]] = []
    mins: np.ndarray | None = None
    for sid, s in enumerate(strands):
        P = s.points
        if P.shape[0] < 2:
            continue
        A = P[:-1]
        B = P[1:]
        mid = 0.5 * (A + B)
        if mins is None:
            mins = mid.min(axis=0)
        else:
            mins = np.minimum(mins, mid.min(axis=0))
        for k in range(mid.shape[0]):
            mids.append(mid[k])
            idx.append((sid, k))
    if mins is None:
        return np.zeros((2,), dtype=np.float64), {}
    mins = mins.astype(np.float64)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for m, key in zip(mids, idx, strict=True):
        gx = int(math.floor(float((m[0] - mins[0]) / cell)))
        gy = int(math.floor(float((m[1] - mins[1]) / cell)))
        buckets.setdefault((gx, gy), []).append(key)
    return mins, buckets


def relax_production(
    strands: list[Strand],
    *,
    min_sep: float,
    poly: Polygon | None = None,
    iters: int = 250,
    max_step_frac: float = 0.25,
) -> None:
    """Production-oriented relaxation focused on whole-curve motion.

    This intentionally avoids per-vertex corrections to prevent zigzagging.
    It only applies per-strand translations derived from detected collisions.
    """

    if not strands:
        return

    cell = float(min_sep)
    max_step = float(max_step_frac) * float(min_sep)

    for it in range(iters):
        _mins, buckets = _build_segment_index(strands, cell=cell)
        if not buckets:
            break

        trans = np.zeros((len(strands), 2), dtype=np.float64)
        processed: set[tuple[int, int, int, int]] = set()
        any_violation = False

        for (gx, gy), segs in buckets.items():
            cand: list[tuple[int, int]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(buckets.get((gx + dx, gy + dy), []))
            for sid, k in segs:
                pts = strands[sid].points
                a = pts[k]
                b = pts[k + 1]
                for sjd, l in cand:
                    if sjd < sid or (sjd == sid and l <= k):
                        continue
                    if sid == sjd:
                        continue
                    key = (sid, k, sjd, l)
                    if key in processed:
                        continue
                    processed.add(key)
                    pts2 = strands[sjd].points
                    c = pts2[l]
                    d = pts2[l + 1]
                    p_cl, q_cl, dist = _segseg_closest(a, b, c, d)
                    if dist >= min_sep:
                        continue
                    any_violation = True
                    u = p_cl - q_cl
                    un = float(np.linalg.norm(u))
                    if un <= 1e-12:
                        continue
                    u = u / un
                    gap = float(min_sep - dist)
                    dv = 0.5 * gap * u
                    trans[sid] += dv
                    trans[sjd] -= dv

        if not any_violation:
            break

        # Damping + clamp.
        trans *= 0.8
        nrm = np.linalg.norm(trans, axis=1)
        for i in range(trans.shape[0]):
            if nrm[i] > max_step:
                trans[i] *= max_step / max(nrm[i], 1e-12)

        for sid, s in enumerate(strands):
            s.points += trans[sid]
            if poly is not None:
                poly_any = cast(Any, poly)
                for i in range(s.points.shape[0]):
                    pt = Point(float(s.points[i, 0]), float(s.points[i, 1]))
                    if bool(poly_any.covers(pt)):
                        continue
                    p = poly_any.exterior.interpolate(poly_any.exterior.project(pt))
                    s.points[i, 0] = float(p.x)
                    s.points[i, 1] = float(p.y)

        if (it % 20) == 0 and iters >= 60:
            print(f"rigid it={it}/{iters}")

    # Stage 2: low-frequency local deformation.
    #
    # If rigid motion alone can't clear all collisions, we resolve remaining overlaps by
    # applying smooth "bump" displacements along each curve's normal field around the
    # collision location. This avoids the pointwise zigzagging you get from per-segment
    # endpoint pushes.
    win = 10
    bump = 0.5 * (1.0 + np.cos(np.linspace(-math.pi, math.pi, 2 * win + 1)))
    bump = bump.astype(np.float64)
    bump /= float(np.sum(bump))

    for it in range(180):
        _mins, buckets = _build_segment_index(strands, cell=cell)
        if not buckets:
            break

        D: list[np.ndarray] = [np.zeros_like(s.points) for s in strands]
        normals: list[np.ndarray] = [_polyline_normals(s.points) for s in strands]

        processed: set[tuple[int, int, int, int]] = set()
        any_violation = False
        for (gx, gy), segs in buckets.items():
            cand: list[tuple[int, int]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(buckets.get((gx + dx, gy + dy), []))
            for sid, k in segs:
                pts = strands[sid].points
                a = pts[k]
                b = pts[k + 1]
                for sjd, l in cand:
                    if sjd < sid or (sjd == sid and l <= k):
                        continue
                    if sid == sjd:
                        continue
                    key = (sid, k, sjd, l)
                    if key in processed:
                        continue
                    processed.add(key)
                    pts2 = strands[sjd].points
                    c = pts2[l]
                    d = pts2[l + 1]
                    p_cl, q_cl, dist = _segseg_closest(a, b, c, d)
                    if dist >= min_sep:
                        continue
                    any_violation = True
                    u = p_cl - q_cl
                    un = float(np.linalg.norm(u))
                    if un <= 1e-12:
                        continue
                    u = u / un
                    gap = float(min_sep - dist)
                    dv = 0.5 * gap * u

                    ni = normals[sid][k]
                    nj = normals[sjd][l]
                    dvi = float(np.dot(dv, ni)) * ni
                    dvj = float(np.dot(-dv, nj)) * nj

                    i0 = max(0, k - win)
                    i1 = min(strands[sid].points.shape[0], k + win + 1)
                    bi = bump[(i0 - (k - win)) : (i1 - (k - win))]
                    D[sid][i0:i1] += bi[:, None] * dvi

                    j0 = max(0, l - win)
                    j1 = min(strands[sjd].points.shape[0], l + win + 1)
                    bj = bump[(j0 - (l - win)) : (j1 - (l - win))]
                    D[sjd][j0:j1] += bj[:, None] * dvj

        if not any_violation:
            break

        for sid, s in enumerate(strands):
            disp = 0.8 * D[sid]
            dn = np.linalg.norm(disp, axis=1)
            max_pt = 0.35 * min_sep
            scale = np.ones_like(dn)
            scale[dn > max_pt] = max_pt / np.maximum(dn[dn > max_pt], 1e-12)
            s.points += disp * scale[:, None]
            if poly is not None:
                poly_any = cast(Any, poly)
                for i in range(s.points.shape[0]):
                    pt = Point(float(s.points[i, 0]), float(s.points[i, 1]))
                    if bool(poly_any.covers(pt)):
                        continue
                    p = poly_any.exterior.interpolate(poly_any.exterior.project(pt))
                    s.points[i, 0] = float(p.x)
                    s.points[i, 1] = float(p.y)

        if (it % 30) == 0:
            print(f"deform it={it}/180")


def verify_relaxed(
    strands: list[Strand], *, min_sep: float, step: float, rmin: float
) -> tuple[float, float]:
    """Return (min_segment_distance, max_turning_angle)."""

    cell = float(min_sep)
    _mins, buckets = _build_segment_index(strands, cell=cell)
    if not buckets:
        return float("inf"), 0.0

    min_dist = float("inf")
    processed: set[tuple[int, int, int, int]] = set()
    exclude_adj = 2
    for (gx, gy), segs in buckets.items():
        cand: list[tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(buckets.get((gx + dx, gy + dy), []))
        for sid, k in segs:
            pts = strands[sid].points
            a = pts[k]
            b = pts[k + 1]
            for sjd, l in cand:
                if sjd < sid or (sjd == sid and l <= k):
                    continue
                if sid == sjd and abs(k - l) <= exclude_adj:
                    continue
                key = (sid, k, sjd, l)
                if key in processed:
                    continue
                processed.add(key)
                pts2 = strands[sjd].points
                c = pts2[l]
                d = pts2[l + 1]
                _, _, dist = _segseg_closest(a, b, c, d)
                if dist < min_dist:
                    min_dist = dist

    max_theta = 0.0
    for s in strands:
        P = s.points
        if P.shape[0] < 3:
            continue
        u = P[1:-1] - P[0:-2]
        v = P[2:] - P[1:-1]
        nu = np.linalg.norm(u, axis=1) + 1e-9
        nv = np.linalg.norm(v, axis=1) + 1e-9
        cos_th = np.sum(u * v, axis=1) / (nu * nv)
        cos_th = np.clip(cos_th, -1.0, 1.0)
        th = np.arccos(cos_th)
        max_theta = max(max_theta, float(np.max(th)))

    _ = step
    _ = rmin
    return min_dist, max_theta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Production-style relaxation for offset lane strands (experiment)."
    )
    ap.add_argument(
        "--input",
        type=str,
        default=str(_resolve_repo_path("data/raw/blob1.svg")),
        help="Input SVG",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(
            _resolve_repo_path("data/processed/blob1_offset_lanes_prodrelax.svg")
        ),
        help="Output SVG",
    )
    ap.add_argument("--min_sep_nm", type=float, default=2.6)
    ap.add_argument("--step_nm", type=float, default=2.6)
    ap.add_argument("--rmin_nm", type=float, default=6.0)
    ap.add_argument("--flat_tol_nm", type=float, default=2.0)
    ap.add_argument("--scale_nm_per_cm", type=float, default=1.0)
    ap.add_argument("--stroke_width", type=str, default="0.1")
    ap.add_argument("--draw_points", action="store_true")
    ap.add_argument("--dot_r_nm", type=float, default=0.2)
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
        t0_nm=0.5 * float(args.step_nm),
        rmin_nm=float(args.rmin_nm),
        min_strand_len_nm=20.0,
        seam_axis="x",
    )
    strands = generate_offset_lane_candidates(
        V, nm_per_unit=float(nm_per_unit), params=params
    )
    print(f"candidates: {len(strands)}")

    poly = Polygon(V)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise ValueError("Polygon is empty")

    step_units = float(args.step_nm) / float(nm_per_unit)
    rmin_units = float(args.rmin_nm) / float(nm_per_unit)
    min_sep_units = float(args.min_sep_nm) / float(nm_per_unit)

    relax_production(strands, min_sep=min_sep_units, poly=poly)

    min_dist, max_theta = verify_relaxed(
        strands, min_sep=min_sep_units, step=step_units, rmin=rmin_units
    )
    theta_max = step_units / rmin_units
    print(
        "verify: "
        f"min_sep_units={min_dist:.6g} (target {min_sep_units:.6g}) "
        f"max_theta={max_theta:.6g} (limit {theta_max:.6g})"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if viewbox is None:
        viewbox = _default_viewbox(V, pad=4.0 * step_units)

    dot_r_units = float(args.dot_r_nm) / float(nm_per_unit)
    _export_preview_svg(
        out_path,
        V=V,
        strands=[s.points.astype(np.float32) for s in strands],
        viewbox=viewbox,
        canvas_size=canvas_size,
        stroke_width=str(args.stroke_width),
        draw_points=bool(args.draw_points),
        dot_r=dot_r_units,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
