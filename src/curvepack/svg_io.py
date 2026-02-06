from __future__ import annotations

import re

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
        print("WARNING: the provided svg shape isn't fully closed")

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


def load_svg_canvas(
    svg_path: str,
) -> tuple[tuple[float, float, float, float] | None, tuple[str, str] | None]:
    """
    Returns (viewbox, canvas_size) if present.
    viewbox: (minx, miny, width, height)
    canvas_size: (width, height) strings with units if provided in the SVG.
    """
    svg_result = svg2paths2(svg_path)
    svg_attributes = svg_result[2] if len(svg_result) > 2 else {}
    viewbox_raw = svg_attributes.get("viewBox") or svg_attributes.get("viewbox")
    viewbox = _parse_viewbox(viewbox_raw)

    width = svg_attributes.get("width")
    height = svg_attributes.get("height")
    canvas_size = (width, height) if width and height else None

    if viewbox is None:
        w = _parse_length(width)
        h = _parse_length(height)
        if w is not None and h is not None:
            viewbox = (0.0, 0.0, w, h)

    return viewbox, canvas_size


def load_svg_scale_nm(svg_path: str, nm_per_cm: float = 1.0) -> float | None:
    """
    Returns nm per SVG user unit if physical units are available.
    Uses width/height units and viewBox for conversion, then applies nm_per_cm.
    """
    svg_result = svg2paths2(svg_path)
    svg_attributes = svg_result[2] if len(svg_result) > 2 else {}
    viewbox_raw = svg_attributes.get("viewBox") or svg_attributes.get("viewbox")
    viewbox = _parse_viewbox(viewbox_raw)

    width = svg_attributes.get("width")
    height = svg_attributes.get("height")

    if viewbox is None:
        w_user = _parse_length(width)
        h_user = _parse_length(height)
        if w_user is not None and h_user is not None:
            viewbox = (0.0, 0.0, w_user, h_user)

    if viewbox is None:
        return None

    width_cm = _parse_length_to_cm(width)
    height_cm = _parse_length_to_cm(height)

    scales: list[float] = []
    if width_cm is not None and viewbox[2] > 0:
        scales.append((width_cm * nm_per_cm) / viewbox[2])
    if height_cm is not None and viewbox[3] > 0:
        scales.append((height_cm * nm_per_cm) / viewbox[3])

    if not scales:
        return None

    return float(sum(scales) / len(scales))


def _parse_viewbox(viewbox_raw: str | None) -> tuple[float, float, float, float] | None:
    if not viewbox_raw:
        return None
    parts = viewbox_raw.replace(",", " ").split()
    if len(parts) != 4:
        return None
    try:
        minx, miny, w, h = (float(p) for p in parts)
    except ValueError:
        return None
    return minx, miny, w, h


def _parse_length(value: str | None) -> float | None:
    parsed = _parse_length_with_unit(value)
    if parsed is None:
        return None
    return parsed[0]


def _parse_length_with_unit(value: str | None) -> tuple[float, str] | None:
    if value is None:
        return None
    if "%" in value:
        return None
    match = re.match(
        r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)\s*$",
        value,
    )
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    return number, unit


def _parse_length_to_cm(value: str | None) -> float | None:
    parsed = _parse_length_with_unit(value)
    if parsed is None:
        return None
    number, unit = parsed
    unit_key = unit.lower()
    if unit_key == "":
        return None
    factor = _UNIT_TO_CM.get(unit_key)
    if factor is None:
        return None
    return number * factor


_UNIT_TO_CM: dict[str, float] = {
    "px": 2.54 / 96.0,
    "pt": 2.54 / 72.0,
    "pc": 2.54 / 6.0,
    "mm": 0.1,
    "cm": 1.0,
    "in": 2.54,
}
