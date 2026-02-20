from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterable, cast

import numpy as np
import svgwrite  # type: ignore[reportMissingTypeStubs]
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon

from src.curvepack.svg_io import (
    load_single_path_polygon,
    load_svg_canvas,
    load_svg_scale_nm,
)


@dataclass(frozen=True)
class VizParams:
    step_nm: float
    t0_nm: float
    rmin_nm: float
    min_strand_len_nm: float
    flat_tol_nm: float
    scale_nm_per_cm: float
    simplify_nm: float
    buffer_resolution: int
    max_lanes: int
    seam_axis: str

    # Visualization
    stroke_width: str
    show_points: bool
    dot_r_nm: float
    max_offsets_to_draw: int

    # Relaxation snapshots
    rigid_iters: int
    rigid_snap_every: int
    deform_iters: int
    deform_snap_every: int


def _resolve_repo_path(rel_path: str) -> Path:
    return Path(__file__).resolve().parents[2] / rel_path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _to_point_list(pts: np.ndarray) -> list[tuple[float, float]]:
    return [(float(p[0]), float(p[1])) for p in pts]


def _export_svg(
    out_path: Path,
    *,
    V: np.ndarray,
    groups: list[tuple[str, Iterable[np.ndarray], dict[str, object]]],
    viewbox: tuple[float, float, float, float],
    canvas_size: tuple[str, str] | None,
    stroke_width: str,
    show_points: bool,
    dot_r: float,
    annotate_endpoints: bool = False,
) -> None:
    dwg = (
        svgwrite.Drawing(str(out_path), profile="tiny")
        if canvas_size is None
        else svgwrite.Drawing(str(out_path), profile="tiny", size=canvas_size)
    )
    dwg.attribs["viewBox"] = f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}"

    dwg.add(
        dwg.polygon(
            points=_to_point_list(V),
            stroke="#111",
            fill="none",
            stroke_width=stroke_width,
            opacity=0.9,
        )
    )

    for gid, polys, style in groups:
        g = dwg.g(id=gid, **style)
        for pts in polys:
            if pts.shape[0] < 2:
                continue
            g.add(
                dwg.polyline(
                    points=_to_point_list(pts),
                    stroke_width=stroke_width,
                    stroke_linecap="round",
                    stroke_linejoin="round",
                )
            )
        dwg.add(g)

    if show_points:
        gp = dwg.g(id="points", fill="#d11", stroke="none", opacity=0.65)
        for _gid, polys, _style in groups:
            for pts in polys:
                for x, y in pts:
                    gp.add(dwg.circle(center=(float(x), float(y)), r=float(dot_r)))
        dwg.add(gp)

    if annotate_endpoints:
        ge = dwg.g(id="endpoints", stroke="none", opacity=0.9)
        for _gid, polys, _style in groups:
            for pts in polys:
                if pts.shape[0] < 2:
                    continue
                # Start (green) and end (blue)
                ge.add(
                    dwg.circle(
                        center=(float(pts[0, 0]), float(pts[0, 1])),
                        r=float(dot_r * 2.5),
                        fill="#1b9e77",
                    )
                )
                ge.add(
                    dwg.circle(
                        center=(float(pts[-1, 0]), float(pts[-1, 1])),
                        r=float(dot_r * 2.5),
                        fill="#377eb8",
                    )
                )
        dwg.add(ge)

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


def _choose_seam_cut_index(pts: np.ndarray, axis: str) -> int:
    if axis not in ("x", "y"):
        raise ValueError("seam_axis must be 'x' or 'y'")
    c = pts.mean(axis=0)
    if axis == "x":
        seam = c[0]
        return int(np.argmin(np.abs(pts[:, 0] - seam)))
    seam = c[1]
    return int(np.argmin(np.abs(pts[:, 1] - seam)))


def _cut_closed_polyline_at_index(pts: np.ndarray, cut_idx: int) -> np.ndarray:
    if pts.shape[0] < 3:
        return pts
    cut_idx = int(cut_idx) % pts.shape[0]
    return np.vstack([pts[cut_idx:], pts[:cut_idx]]).astype(np.float32)


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


def _ensure_single_polygon(poly: Polygon) -> Polygon:
    """Ensure we have a single Polygon; pick the largest if buffering creates MultiPolygon."""
    poly_any = cast(Any, poly)
    if getattr(poly_any, "geom_type", None) == "MultiPolygon":
        return cast(Polygon, max(poly_any.geoms, key=lambda g: g.area))
    return poly


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
    polylines: list[np.ndarray], cell: float
) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int]]]]:
    mids: list[np.ndarray] = []
    idx: list[tuple[int, int]] = []
    mins: np.ndarray | None = None
    for sid, P in enumerate(polylines):
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


def _relax_with_snapshots(
    polylines: list[np.ndarray],
    *,
    min_sep: float,
    poly: Polygon,
    rigid_iters: int,
    rigid_snap_every: int,
    deform_iters: int,
    deform_snap_every: int,
    snapshot_cb: Callable[[str, int, list[np.ndarray]], None],
) -> list[np.ndarray]:
    cell = float(min_sep)
    max_step = 0.25 * min_sep
    P_list = [P.astype(np.float64).copy() for P in polylines]

    def clamp_inside() -> None:
        poly_any = cast(Any, poly)
        for pts in P_list:
            for i in range(pts.shape[0]):
                pt = Point(float(pts[i, 0]), float(pts[i, 1]))
                if bool(poly_any.covers(pt)):
                    continue
                q = poly_any.exterior.interpolate(poly_any.exterior.project(pt))
                pts[i, 0] = float(q.x)
                pts[i, 1] = float(q.y)

    snapshot_cb("relax_start", 0, [P.astype(np.float32) for P in P_list])

    # Stage 1: rigid packing.
    for it in range(rigid_iters):
        _mins, buckets = _build_segment_index(P_list, cell=cell)
        if not buckets:
            break
        trans = np.zeros((len(P_list), 2), dtype=np.float64)
        processed: set[tuple[int, int, int, int]] = set()
        any_violation = False
        for (gx, gy), segs in buckets.items():
            cand: list[tuple[int, int]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(buckets.get((gx + dx, gy + dy), []))
            for sid, k in segs:
                pts = P_list[sid]
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
                    pts2 = P_list[sjd]
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
        trans *= 0.8
        nrm = np.linalg.norm(trans, axis=1)
        for i in range(trans.shape[0]):
            if nrm[i] > max_step:
                trans[i] *= max_step / max(nrm[i], 1e-12)
        for sid, _pts in enumerate(P_list):
            P_list[sid] += trans[sid]
        clamp_inside()
        if (it % max(1, rigid_snap_every)) == 0:
            snapshot_cb("rigid", it, [P.astype(np.float32) for P in P_list])

    snapshot_cb("rigid_end", rigid_iters, [P.astype(np.float32) for P in P_list])

    # Stage 2: smooth deformation.
    win = 10
    bump = 0.5 * (1.0 + np.cos(np.linspace(-math.pi, math.pi, 2 * win + 1)))
    bump = bump.astype(np.float64)
    bump /= float(np.sum(bump))

    for it in range(deform_iters):
        _mins, buckets = _build_segment_index(P_list, cell=cell)
        if not buckets:
            break
        D = [np.zeros_like(P) for P in P_list]
        normals = [_polyline_normals(P) for P in P_list]
        processed = set()
        any_violation = False
        for (gx, gy), segs in buckets.items():
            cand: list[tuple[int, int]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand.extend(buckets.get((gx + dx, gy + dy), []))
            for sid, k in segs:
                pts = P_list[sid]
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
                    pts2 = P_list[sjd]
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
                    i1 = min(pts.shape[0], k + win + 1)
                    bi = bump[(i0 - (k - win)) : (i1 - (k - win))]
                    D[sid][i0:i1] += bi[:, None] * dvi

                    j0 = max(0, l - win)
                    j1 = min(pts2.shape[0], l + win + 1)
                    bj = bump[(j0 - (l - win)) : (j1 - (l - win))]
                    D[sjd][j0:j1] += bj[:, None] * dvj

        if not any_violation:
            break

        for sid, pts in enumerate(P_list):
            disp = 0.8 * D[sid]
            dn = np.linalg.norm(disp, axis=1)
            max_pt = 0.35 * min_sep
            scale = np.ones_like(dn)
            scale[dn > max_pt] = max_pt / np.maximum(dn[dn > max_pt], 1e-12)
            P_list[sid] += disp * scale[:, None]
        clamp_inside()
        if (it % max(1, deform_snap_every)) == 0:
            snapshot_cb("deform", it, [P.astype(np.float32) for P in P_list])

    snapshot_cb("relax_end", deform_iters, [P.astype(np.float32) for P in P_list])
    return [P.astype(np.float32) for P in P_list]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate step-by-step SVGs for lane generation + relaxation."
    )
    ap.add_argument(
        "--input",
        type=str,
        default=str(_resolve_repo_path("data/raw/blob1.svg")),
        help="Input SVG path",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_resolve_repo_path("data/processed/blob1_lane_steps")),
        help="Output directory (folder of SVG snapshots)",
    )
    args = ap.parse_args()

    params = VizParams(
        step_nm=2.6,
        t0_nm=1.3,
        rmin_nm=6.0,
        min_strand_len_nm=20.0,
        flat_tol_nm=2.0,
        scale_nm_per_cm=1.0,
        simplify_nm=0.5,
        buffer_resolution=8,
        max_lanes=200,
        seam_axis="x",
        stroke_width="0.1",
        show_points=False,
        dot_r_nm=0.2,
        max_offsets_to_draw=18,
        rigid_iters=250,
        rigid_snap_every=25,
        deform_iters=180,
        deform_snap_every=30,
    )

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input SVG not found: {in_path}")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    viewbox, canvas_size = load_svg_canvas(str(in_path))
    nm_per_unit = load_svg_scale_nm(
        str(in_path), nm_per_cm=float(params.scale_nm_per_cm)
    )
    if nm_per_unit is None or nm_per_unit <= 0:
        nm_per_unit = 1.0
        print("WARNING: svg has no physical units; assuming 1 nm per SVG unit")
    else:
        print(
            "svg_scale "
            f"nm_per_unit={nm_per_unit:.6g} nm_per_cm={float(params.scale_nm_per_cm):.6g}"
        )

    flat_tol_units = float(params.flat_tol_nm) / float(nm_per_unit)
    V = load_single_path_polygon(str(in_path), flat_tol=flat_tol_units)

    step_units = float(params.step_nm) / float(nm_per_unit)
    t0_units = float(params.t0_nm) / float(nm_per_unit)
    rmin_units = float(params.rmin_nm) / float(nm_per_unit)
    min_len_units = float(params.min_strand_len_nm) / float(nm_per_unit)
    dot_r_units = float(params.dot_r_nm) / float(nm_per_unit)

    if viewbox is None:
        viewbox = _default_viewbox(V, pad=4.0 * step_units)

    poly = Polygon(V)
    if not poly.is_valid:
        fixed = poly.buffer(0)
        fixed_any = cast(Any, fixed)
        if getattr(fixed_any, "geom_type", None) == "MultiPolygon":
            poly = cast(Polygon, max(fixed_any.geoms, key=lambda g: g.area))
        else:
            poly = cast(Polygon, fixed_any)
    if poly.is_empty:
        raise ValueError("Polygon is empty after fixing")
    poly = _ensure_single_polygon(poly)
    if params.simplify_nm > 0:
        simp_units = float(params.simplify_nm) / float(nm_per_unit)
        simplified = poly.simplify(simp_units, preserve_topology=True)
        simplified_any = cast(Any, simplified)
        if getattr(simplified_any, "geom_type", None) == "MultiPolygon":
            poly = cast(Polygon, max(simplified_any.geoms, key=lambda g: g.area))
        else:
            poly = cast(Polygon, simplified_any)
        if poly.is_empty:
            raise ValueError("Polygon is empty after simplify")

    # 00: outline
    _export_svg(
        out_dir / "00_outline.svg",
        V=V,
        groups=[],
        viewbox=viewbox,
        canvas_size=canvas_size,
        stroke_width=params.stroke_width,
        show_points=False,
        dot_r=dot_r_units,
    )

    # 01: inset offsets (draw multiple rings across k)
    all_rings: list[np.ndarray] = []
    for k in range(params.max_lanes):
        if len(all_rings) >= params.max_offsets_to_draw:
            break
        t = t0_units + k * step_units
        inner = poly.buffer(-t, resolution=int(params.buffer_resolution))
        if getattr(inner, "is_empty", False):
            break
        rings = _extract_rings(inner)
        for ring in rings:
            if ring.shape[0] >= 2 and np.linalg.norm(ring[0] - ring[-1]) < 1e-6:
                ring = ring[:-1]
            if ring.shape[0] < 3:
                continue
            all_rings.append(ring.astype(np.float32))

    _export_svg(
        out_dir / "01_offsets_raw.svg",
        V=V,
        groups=[
            (
                "offset_rings",
                all_rings,
                {"fill": "none", "stroke": "#d11", "opacity": 0.45},
            )
        ],
        viewbox=viewbox,
        canvas_size=canvas_size,
        stroke_width=params.stroke_width,
        show_points=False,
        dot_r=dot_r_units,
    )

    # 02: resampled rings
    rings_rs = [_arclength_resample_closed(r, step=step_units) for r in all_rings]
    _export_svg(
        out_dir / "02_offsets_resampled.svg",
        V=V,
        groups=[
            (
                "offset_resampled",
                rings_rs,
                {"fill": "none", "stroke": "#d11", "opacity": 0.55},
            )
        ],
        viewbox=viewbox,
        canvas_size=canvas_size,
        stroke_width=params.stroke_width,
        show_points=params.show_points,
        dot_r=dot_r_units,
    )

    # 03: seam cut (open polylines)
    seam_open: list[np.ndarray] = []
    for r in rings_rs:
        cut_idx = _choose_seam_cut_index(r, axis=params.seam_axis)
        seam_open.append(_cut_closed_polyline_at_index(r, cut_idx))
    _export_svg(
        out_dir / "03_seam_cut.svg",
        V=V,
        groups=[
            (
                "seam_open",
                seam_open,
                {"fill": "none", "stroke": "#d11", "opacity": 0.7},
            )
        ],
        viewbox=viewbox,
        canvas_size=canvas_size,
        stroke_width=params.stroke_width,
        show_points=params.show_points,
        dot_r=dot_r_units,
        annotate_endpoints=True,
    )

    # 04: curvature pruning
    kappa_max = 1.0 / float(rmin_units)
    min_points = max(4, int(math.ceil(min_len_units / max(step_units, 1e-9))))
    pruned: list[np.ndarray] = []
    for s in seam_open:
        pruned.extend(
            _split_on_curvature_violations(
                s, kappa_max=kappa_max, min_points=min_points, guard=2
            )
        )
    _export_svg(
        out_dir / "04_pruned.svg",
        V=V,
        groups=[
            (
                "pruned",
                pruned,
                {"fill": "none", "stroke": "#d11", "opacity": 0.8},
            )
        ],
        viewbox=viewbox,
        canvas_size=canvas_size,
        stroke_width=params.stroke_width,
        show_points=params.show_points,
        dot_r=dot_r_units,
        annotate_endpoints=True,
    )

    # 05+: relaxation snapshots
    min_sep_units = step_units  # by design, centerline spacing equals step

    def snapshot(stage: str, it: int, curves: list[np.ndarray]) -> None:
        fname = f"05_relax_{stage}_{it:04d}.svg"
        _export_svg(
            out_dir / fname,
            V=V,
            groups=[
                (
                    "relaxed",
                    curves,
                    {"fill": "none", "stroke": "#d11", "opacity": 0.85},
                )
            ],
            viewbox=viewbox,
            canvas_size=canvas_size,
            stroke_width=params.stroke_width,
            show_points=False,
            dot_r=dot_r_units,
        )

    _relax_with_snapshots(
        pruned,
        min_sep=min_sep_units,
        poly=poly,
        rigid_iters=params.rigid_iters,
        rigid_snap_every=params.rigid_snap_every,
        deform_iters=params.deform_iters,
        deform_snap_every=params.deform_snap_every,
        snapshot_cb=snapshot,
    )

    # Write a tiny index for Obsidian.
    index = out_dir / "INDEX.md"
    index.write_text(
        "# Lane + Relaxation Steps\n\n"
        "Open the SVGs in this folder in filename order.\n\n"
        "- 00: outline\n"
        "- 01: raw inward offsets (rings)\n"
        "- 02: offsets resampled at 2.6nm\n"
        "- 03: seam cut (open strands)\n"
        "- 04: curvature-pruned strands\n"
        "- 05: relaxation snapshots (rigid + deform stages)\n",
        encoding="utf-8",
    )

    print(f"Wrote snapshots to: {out_dir}")


if __name__ == "__main__":
    main()
