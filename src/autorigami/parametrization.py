from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
            raise ValueError("piecewise Hermite curve must contain at least one segment")


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


def polyline_to_cubic_bezier_chain(polyline: Polyline) -> PiecewiseBezier:
    """Convert a polyline to a cubic Bezier chain with one segment per edge."""

    pts = polyline.points
    tangents = np.empty_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents[1:-1] = 0.5 * (pts[2:] - pts[:-2])

    segments = np.empty((pts.shape[0] - 1, 4, 3), dtype=np.float64)
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        ds = float(np.linalg.norm(p1 - p0))

        tangent0_norm = float(np.linalg.norm(tangents[i]))
        tangent1_norm = float(np.linalg.norm(tangents[i + 1]))
        if ds == 0.0 or tangent0_norm == 0.0 or tangent1_norm == 0.0:
            t0 = np.zeros(3, dtype=np.float64)
            t1 = np.zeros(3, dtype=np.float64)
        else:
            t0 = tangents[i] / tangent0_norm
            t1 = tangents[i + 1] / tangent1_norm

        segments[i, 0, :] = p0
        segments[i, 1, :] = p0 + (ds / 3.0) * t0
        segments[i, 2, :] = p1 - (ds / 3.0) * t1
        segments[i, 3, :] = p1

    return PiecewiseBezier(segments=segments)


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
