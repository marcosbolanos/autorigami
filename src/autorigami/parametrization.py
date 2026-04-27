from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, PPoly, splprep
from autorigami.types import FloatArray


@dataclass(frozen=True, slots=True)
class Polyline:
    """A polyline in R^3 represented by ordered points."""

    points: FloatArray

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if self.points.shape[0] < 2:
            raise ValueError("polyline must contain at least two points")


@dataclass(frozen=True, slots=True)
class PiecewiseHermite:
    """Piecewise cubic Hermite curve in R^3 using nodal positions and tangents."""

    points: FloatArray
    tangents: FloatArray

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if self.tangents.ndim != 2 or self.tangents.shape[1] != 3:
            raise ValueError("tangents must have shape (N, 3)")
        if self.points.shape != self.tangents.shape:
            raise ValueError("points and tangents must have the same shape")
        if self.points.shape[0] < 2:
            raise ValueError(
                "piecewise Hermite curve must contain at least one segment"
            )


@dataclass(frozen=True, slots=True)
class PiecewiseBezier:
    """Chain of cubic Bezier segments in R^3."""

    segments: FloatArray

    def __post_init__(self) -> None:
        if self.segments.ndim != 3 or self.segments.shape[1:] != (4, 3):
            raise ValueError("segments must have shape (M, 4, 3)")
        if self.segments.shape[0] < 1:
            raise ValueError("need at least one Bezier segment")


def piecewise_hermite_to_bezier(hermite: PiecewiseHermite) -> PiecewiseBezier:
    """Convert a piecewise cubic Hermite curve to an exactly equivalent Bezier chain.

    For each segment [P_i, P_{i+1}] with endpoint tangents [T_i, T_{i+1}] on
    unit parameter interval [0, 1], the equivalent cubic Bezier control points are:
    B0 = P_i
    B1 = P_i + T_i / 3
    B2 = P_{i+1} - T_{i+1} / 3
    B3 = P_{i+1}
    """

    p0 = hermite.points[:-1]
    p1 = hermite.points[1:]
    m0 = hermite.tangents[:-1]
    m1 = hermite.tangents[1:]

    segments = np.empty((p0.shape[0], 4, 3), dtype=np.float64)
    segments[:, 0, :] = p0
    segments[:, 1, :] = p0 + (m0 / 3.0)
    segments[:, 2, :] = p1 - (m1 / 3.0)
    segments[:, 3, :] = p1

    return PiecewiseBezier(segments=segments)


def _cubic_bezier_eval(control: FloatArray, t: FloatArray) -> FloatArray:
    t_col = t[:, None]
    one_minus_t = 1.0 - t_col
    return (
        (one_minus_t**3) * control[0]
        + 3.0 * (one_minus_t**2) * t_col * control[1]
        + 3.0 * one_minus_t * (t_col**2) * control[2]
        + (t_col**3) * control[3]
    )


def sample_cubic_bezier_chain(beziers: PiecewiseBezier, num_samples: int) -> Polyline:
    """Sample points from a cubic Bezier chain."""

    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")

    curves = beziers.segments
    lengths = np.linalg.norm(curves[:, 3, :] - curves[:, 0, :], axis=1)
    weights = lengths / max(float(lengths.sum()), 1e-12)
    counts = np.maximum(2, np.floor(weights * num_samples).astype(int))

    diff = int(num_samples - counts.sum())
    if diff > 0:
        order = np.argsort(-weights)
        for idx in order[:diff]:
            counts[idx] += 1
    elif diff < 0:
        order = np.argsort(weights)
        for idx in order:
            if diff == 0:
                break
            removable = min(counts[idx] - 2, -diff)
            if removable > 0:
                counts[idx] -= removable
                diff += removable

    sampled: list[FloatArray] = []
    for i, (curve, n) in enumerate(zip(curves, counts)):
        ts = np.linspace(0.0, 1.0, int(n), endpoint=True, dtype=np.float64)
        seg_points = _cubic_bezier_eval(curve, ts)
        if i > 0:
            seg_points = seg_points[1:]
        sampled.append(seg_points)

    out = np.vstack(sampled)

    if out.shape[0] > num_samples:
        out = out[:num_samples]
    elif out.shape[0] < num_samples:
        pad = np.repeat(out[-1][None, :], num_samples - out.shape[0], axis=0)
        out = np.vstack([out, pad])

    return Polyline(points=out)


def fit_parametric_bspline_from_polyline(
    polyline: Polyline,
    smoothing: float,
    degree: int = 3,
) -> tuple[BSpline, BSpline, BSpline]:
    """Fit a parametric B-spline p(u) to a polyline."""

    points = np.asarray(polyline.points, dtype=np.float64)
    if points.shape[0] < 2:
        raise ValueError("polyline must contain at least two points")
    if degree < 1 or degree > 5:
        raise ValueError("degree must be in [1, 5]")
    if smoothing < 0.0:
        raise ValueError("smoothing must be >= 0")

    edge = points[1:] - points[:-1]
    chord = np.linalg.norm(edge, axis=1)
    cumulative = np.zeros(points.shape[0], dtype=np.float64)
    cumulative[1:] = np.cumsum(chord)
    total = float(cumulative[-1])
    if total <= 1e-12:
        raise ValueError("polyline has near-zero total length")
    u = cumulative / total

    k = min(degree, points.shape[0] - 1)
    tck, _ = splprep(
        [points[:, 0], points[:, 1], points[:, 2]],
        u=u,
        s=smoothing,
        k=k,
        per=0,
    )
    knots, coeffs, deg = tck
    return (
        BSpline(knots, coeffs[0], deg, extrapolate=False),
        BSpline(knots, coeffs[1], deg, extrapolate=False),
        BSpline(knots, coeffs[2], deg, extrapolate=False),
    )


def sample_parametric_bspline(
    splines: tuple[BSpline, BSpline, BSpline],
    num_samples: int,
) -> Polyline:
    """Sample a parametric B-spline tuple."""
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")
    sx, sy, sz = splines
    u0 = float(max(sx.t[sx.k], sy.t[sy.k], sz.t[sz.k]))
    u1 = float(min(sx.t[-sx.k - 1], sy.t[-sy.k - 1], sz.t[-sz.k - 1]))
    if not np.isfinite(u0) or not np.isfinite(u1) or u1 <= u0:
        raise ValueError("invalid spline parameter domain")
    u = np.linspace(u0, u1, num_samples, dtype=np.float64)
    pts = np.column_stack((sx(u), sy(u), sz(u))).astype(np.float64, copy=False)
    return Polyline(points=pts)


def cubic_bspline_to_piecewise_bezier(
    splines: tuple[BSpline, BSpline, BSpline],
) -> PiecewiseBezier:
    """Exactly convert a cubic parametric B-spline to a Bezier chain."""
    bx, by, bz = splines
    if bx.k != 3 or by.k != 3 or bz.k != 3:
        raise ValueError("all splines must be cubic")
    if not (np.allclose(bx.t, by.t) and np.allclose(bx.t, bz.t)):
        raise ValueError("parametric spline knot vectors do not match")

    px = PPoly.from_spline(bx)
    py = PPoly.from_spline(by)
    pz = PPoly.from_spline(bz)
    if not (np.allclose(px.x, py.x) and np.allclose(px.x, pz.x)):
        raise ValueError("parametric polynomial breakpoints do not match")
    if px.c.shape[0] != 4 or py.c.shape[0] != 4 or pz.c.shape[0] != 4:
        raise ValueError("expected cubic piecewise-polynomial coefficients")

    breaks = px.x
    segment_controls: list[np.ndarray] = []
    for i in range(breaks.shape[0] - 1):
        h = float(breaks[i + 1] - breaks[i])
        if h <= 1e-14:
            continue

        # p(u) = A*u^3 + B*u^2 + C*u + D on u in [0,1]
        # where x = x_i + h*u and p(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + d
        a = np.array([px.c[0, i], py.c[0, i], pz.c[0, i]], dtype=np.float64)
        b = np.array([px.c[1, i], py.c[1, i], pz.c[1, i]], dtype=np.float64)
        c = np.array([px.c[2, i], py.c[2, i], pz.c[2, i]], dtype=np.float64)
        d = np.array([px.c[3, i], py.c[3, i], pz.c[3, i]], dtype=np.float64)
        A = a * (h**3)
        B = b * (h**2)
        C = c * h
        D = d

        control = np.empty((4, 3), dtype=np.float64)
        control[0, :] = D
        control[1, :] = D + C / 3.0
        control[2, :] = D + (2.0 / 3.0) * C + B / 3.0
        control[3, :] = D + C + B + A
        segment_controls.append(control)

    if not segment_controls:
        raise ValueError("no non-degenerate spline intervals available for Bezier conversion")

    segments = np.stack(segment_controls, axis=0)

    return PiecewiseBezier(segments=segments)


def _estimate_polyline_curvature_max(points: FloatArray) -> float:
    if points.shape[0] < 3:
        return 0.0
    edges = points[1:] - points[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    tangents = edges / np.maximum(lengths[:, None], 1e-12)
    delta_tangent = tangents[1:] - tangents[:-1]
    ds = 0.5 * (lengths[1:] + lengths[:-1])
    curvature = np.linalg.norm(delta_tangent, axis=1) / np.maximum(ds, 1e-12)
    return float(np.max(curvature))


def fit_constrained_bezier_from_polyline(
    polyline: Polyline,
    max_handle_fraction: float = 0.08,
    max_curvature: float | None = None,
    max_shrink_iters: int = 10,
    curvature_samples_per_segment: int = 12,
) -> PiecewiseBezier:
    """Fit a bounded C1 Bezier chain to a polyline with monotonicity guards."""

    points = np.asarray(polyline.points, dtype=np.float64)
    if points.shape[0] < 2:
        raise ValueError("polyline must contain at least two points")
    if max_handle_fraction <= 0.0:
        raise ValueError("max_handle_fraction must be > 0")
    if max_shrink_iters < 0:
        raise ValueError("max_shrink_iters must be >= 0")
    if curvature_samples_per_segment < 2:
        raise ValueError("curvature_samples_per_segment must be >= 2")

    edges = points[1:] - points[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    edge_dirs = edges / np.maximum(edge_lengths[:, None], 1e-12)
    node_count = points.shape[0]
    segment_count = node_count - 1

    node_dirs = np.zeros((node_count, 3), dtype=np.float64)
    node_dirs[0] = edge_dirs[0]
    node_dirs[-1] = edge_dirs[-1]
    for i in range(1, node_count - 1):
        candidate = edge_dirs[i - 1] + edge_dirs[i]
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm < 1e-12:
            candidate = edge_dirs[i]
        else:
            candidate = candidate / candidate_norm
        if np.dot(candidate, edge_dirs[i - 1]) <= 0.0 or np.dot(candidate, edge_dirs[i]) <= 0.0:
            candidate = edge_dirs[i]
        node_dirs[i] = candidate

    node_handle_base = np.zeros(node_count, dtype=np.float64)
    node_handle_base[0] = max_handle_fraction * edge_lengths[0]
    node_handle_base[-1] = max_handle_fraction * edge_lengths[-1]
    for i in range(1, node_count - 1):
        node_handle_base[i] = max_handle_fraction * min(edge_lengths[i - 1], edge_lengths[i])

    def build_segments(scale: float) -> FloatArray:
        node_handle = node_handle_base * scale
        segments = np.empty((segment_count, 4, 3), dtype=np.float64)
        min_forward_fraction = 0.01
        max_perp_fraction = 0.05

        for i in range(segment_count):
            p0 = points[i]
            p1 = points[i + 1]
            chord = edge_dirs[i]
            length = edge_lengths[i]
            h0 = node_dirs[i] * node_handle[i]
            h1 = node_dirs[i + 1] * node_handle[i + 1]

            # Keep handles monotone along the segment chord and limit lateral drift.
            for handle in (h0, h1):
                forward = float(np.dot(handle, chord))
                min_forward = min_forward_fraction * length
                if forward < min_forward:
                    handle += chord * (min_forward - forward)
                perp = handle - chord * float(np.dot(handle, chord))
                perp_norm = float(np.linalg.norm(perp))
                max_perp = max_perp_fraction * length
                if perp_norm > max_perp and perp_norm > 1e-12:
                    handle -= perp * (1.0 - (max_perp / perp_norm))

            segments[i, 0, :] = p0
            segments[i, 1, :] = p0 + h0
            segments[i, 2, :] = p1 - h1
            segments[i, 3, :] = p1
        return segments

    scale = 1.0
    best_segments = build_segments(scale)
    if max_curvature is not None and max_curvature > 0.0:
        for _ in range(max_shrink_iters + 1):
            candidate = PiecewiseBezier(segments=best_segments)
            sampled = sample_cubic_bezier_chain(
                candidate,
                num_samples=max(
                    points.shape[0] * curvature_samples_per_segment,
                    points.shape[0],
                ),
            )
            curvature_max = _estimate_polyline_curvature_max(sampled.points)
            if curvature_max <= max_curvature:
                break
            scale *= 0.7
            best_segments = build_segments(scale)

    return PiecewiseBezier(segments=best_segments)
