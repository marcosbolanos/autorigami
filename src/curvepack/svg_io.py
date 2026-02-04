from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped
from svgpathtools import svg2paths2, Path  # type: ignore[reportMissingTypeStubs]


@jaxtyped(typechecker=beartype)
def load_single_path_polygon(
    svg_path: str,
    flat_tol: float = 1.0,
) -> Float[np.ndarray, "N 2"]:
    """
    Returns polygon vertices V: (N,2) approximating the outline.
    Assumes one continuous closed object (one path). No holes.
    Coordinates are in SVG user units.
    """
    paths = svg2paths2(svg_path)[0]
    if len(paths) == 0:
        raise ValueError("No <path> found in SVG.")
    p: Path = paths[0]

    # Ensure closed
    if not p.isclosed():
        p = Path(*p)  # copy
        p.append(p[0].reversed())  # crude fallback; better: require closed
        # In practice, you should fix the SVG; for now, allow but warn.

    # Flatten path to polyline with specified tolerance.
    # svgpathtools provides "poly" sampling via continuous discretization.
    # We'll adaptively sample each segment by length / tol heuristic.
    pts: list[tuple[float, float]] = []
    for seg in p:
        # Sample a segment into points. Heuristic: number of samples based on seg length / tol.
        L = max(float(seg.length(error=1e-3)), 1e-6)
        n = max(4, int(np.ceil(L / max(flat_tol, 1e-6))))
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        for t in ts:
            z = seg.point(t)
            pts.append((z.real, z.imag))
    # Add final point to close
    z = p[-1].point(1.0)
    pts.append((z.real, z.imag))

    vertices = np.asarray(pts, dtype=np.float32)
    # Remove near-duplicates
    keep: list[int] = [0]
    for i in range(1, len(vertices)):
        if np.linalg.norm(vertices[i] - vertices[keep[-1]]) > (flat_tol * 0.25):
            keep.append(i)
    vertices = vertices[keep]
    # Ensure last equals first
    if np.linalg.norm(vertices[0] - vertices[-1]) > 1e-6:
        vertices = np.vstack([vertices, vertices[0]])
    return vertices
