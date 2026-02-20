from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def polyline_to_cubic_beziers(
    points: Float[np.ndarray, "N 2"],
    *,
    handle_scale: float = 1.0,
    max_handle_ratio: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Convert an open polyline into cubic Bezier segments.

    The output is an interpolating spline: each segment starts/ends at consecutive points.

    This is Catmull-Rom-style conversion with conservative handle clamping:
    - `handle_scale` controls smoothing (0 -> straight segments, 1 -> full Catmull-Rom).
    - `max_handle_ratio` caps each handle length to a fraction of its segment length.

    Parameters
    - points: (N,2) polyline vertices in world coords.
    - handle_scale: [0, +inf) smoothing multiplier; use <= 1 for safety.
    - max_handle_ratio: cap for |c1-p0| and |p3-c2| relative to segment length.

    Returns
    - segs: list of (p0, c1, c2, p3) float32 arrays, one per polyline edge.
    """

    P = np.asarray(points, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    if P.shape[0] < 2:
        return []
    if not np.isfinite(P).all():
        raise ValueError("points contains non-finite coordinates")
    if not np.isfinite(handle_scale) or handle_scale < 0:
        raise ValueError("handle_scale must be finite and >= 0")
    if not np.isfinite(max_handle_ratio) or max_handle_ratio < 0:
        raise ValueError("max_handle_ratio must be finite and >= 0")

    # Vertex tangents (Catmull-Rom / Hermite style).
    #
    # IMPORTANT: we clamp *per-vertex tangents*, not per-segment handles.
    # That preserves C1 continuity at joints even when we cap handle lengths.
    N = P.shape[0]
    m = np.zeros_like(P, dtype=np.float32)
    if N == 2:
        m[0] = P[1] - P[0]
        m[1] = P[1] - P[0]
    else:
        m[0] = P[1] - P[0]
        m[-1] = P[-1] - P[-2]
        m[1:-1] = 0.5 * (P[2:] - P[:-2])

    s = float(handle_scale)
    if s > 0:
        # Clamp tangent magnitudes so the resulting handle lengths are bounded for
        # both adjacent segments.
        seg_len = np.linalg.norm(P[1:] - P[:-1], axis=1).astype(np.float32)
        for i in range(N):
            L_left = float(seg_len[i - 1]) if i > 0 else float(seg_len[0])
            L_right = float(seg_len[i]) if i < (N - 1) else float(seg_len[-1])
            L = min(L_left, L_right)
            if not np.isfinite(L) or L <= 1e-12:
                m[i] = 0.0
                continue

            max_handle = float(max_handle_ratio) * L
            max_tan = (3.0 / s) * max_handle
            tn = float(np.linalg.norm(m[i]))
            if tn > max_tan and tn > 1e-12:
                m[i] *= float(max_tan / tn)

            # Prevent "backward" tangents relative to adjacent segment directions.
            # This reduces overshoot and avoids kinks in monotone-ish polylines.
            if i > 0:
                d0 = P[i] - P[i - 1]
                d0n2 = float(np.dot(d0, d0))
                if d0n2 > 1e-12:
                    dot0 = float(np.dot(m[i], d0))
                    if dot0 < 0.0:
                        m[i] -= (dot0 / d0n2) * d0
            if i < (N - 1):
                d1 = P[i + 1] - P[i]
                d1n2 = float(np.dot(d1, d1))
                if d1n2 > 1e-12:
                    dot1 = float(np.dot(m[i], d1))
                    if dot1 < 0.0:
                        m[i] -= (dot1 / d1n2) * d1

    segs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(N - 1):
        p0 = P[i]
        p3 = P[i + 1]
        if np.linalg.norm(p3 - p0) <= 1e-12:
            # Degenerate edge.
            c1 = p0.copy()
            c2 = p3.copy()
            segs.append((p0, c1, c2, p3))
            continue

        if s <= 0.0:
            c1 = p0.copy()
            c2 = p3.copy()
            segs.append((p0.astype(np.float32), c1, c2, p3.astype(np.float32)))
            continue

        c1 = (p0 + (s / 3.0) * m[i]).astype(np.float32)
        c2 = (p3 - (s / 3.0) * m[i + 1]).astype(np.float32)
        segs.append((p0.astype(np.float32), c1, c2, p3.astype(np.float32)))

    return segs


def beziers_to_svg_path_d(
    segs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    precision: int = 3,
) -> str:
    """Build an SVG path 'd' string from cubic Bezier segments."""

    if not segs:
        return ""
    fmt = f".{{:d}}f".format(int(precision))

    def f(x: float) -> str:
        return format(float(x), fmt)

    p0 = segs[0][0]
    parts = [f"M {f(p0[0])},{f(p0[1])}"]
    for _p0, c1, c2, p3 in segs:
        parts.append(
            f"C {f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} {f(p3[0])},{f(p3[1])}"
        )
    return " ".join(parts)


@jaxtyped(typechecker=beartype)
def estimate_max_turning_angle(
    points: Float[np.ndarray, "N 2"],
    eps: float = 1e-9,
) -> float:
    """Return the maximum discrete turning angle (radians) along an open polyline."""

    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    if P.shape[0] < 3:
        return 0.0
    u = P[1:-1] - P[0:-2]
    v = P[2:] - P[1:-1]
    nu = np.linalg.norm(u, axis=1) + eps
    nv = np.linalg.norm(v, axis=1) + eps
    cos_th = np.sum(u * v, axis=1) / (nu * nv)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    return float(np.max(th))
