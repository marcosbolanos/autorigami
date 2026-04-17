from __future__ import annotations

import numpy as np


def _cubic_bezier_eval(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    one_minus_t = 1.0 - t
    return (
        (one_minus_t**3) * control[0]
        + 3.0 * (one_minus_t**2) * t * control[1]
        + 3.0 * one_minus_t * (t**2) * control[2]
        + (t**3) * control[3]
    )


def polyline_to_cubic_bezier_chain(points: np.ndarray) -> np.ndarray:
    """Convert a polyline to a fine cubic Bezier chain.

    Uses one cubic segment per consecutive point pair (finest practical chain)
    with Hermite tangents converted to Bezier control points.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if pts.shape[0] < 2:
        raise ValueError("need at least 2 points")

    tangents = np.empty_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    tangents[1:-1] = 0.5 * (pts[2:] - pts[:-2])

    beziers = np.empty((pts.shape[0] - 1, 4, 3), dtype=float)
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        ds = float(np.linalg.norm(p1 - p0))
        if ds <= 1e-12:
            t0 = np.zeros(3, dtype=float)
            t1 = np.zeros(3, dtype=float)
        else:
            t0 = tangents[i] / (np.linalg.norm(tangents[i]) + 1e-12)
            t1 = tangents[i + 1] / (np.linalg.norm(tangents[i + 1]) + 1e-12)

        b0 = p0
        b1 = p0 + (ds / 3.0) * t0
        b2 = p1 - (ds / 3.0) * t1
        b3 = p1
        beziers[i] = np.array([b0, b1, b2, b3], dtype=float)

    return beziers


def sample_cubic_bezier_chain(beziers: np.ndarray, num_samples: int) -> np.ndarray:
    """Sample points from a cubic Bezier chain."""
    curves = np.asarray(beziers, dtype=float)
    if curves.ndim != 3 or curves.shape[1:] != (4, 3):
        raise ValueError("beziers must have shape (M, 4, 3)")
    if curves.shape[0] < 1:
        raise ValueError("need at least one Bezier segment")
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")

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

    sampled: list[np.ndarray] = []
    for i, (curve, n) in enumerate(zip(curves, counts)):
        ts = np.linspace(0.0, 1.0, int(n), endpoint=True)
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
    return out
