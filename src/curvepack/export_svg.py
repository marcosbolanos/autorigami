from __future__ import annotations

from typing import TypeAlias

import numpy as np
import svgwrite  # type: ignore[reportMissingTypeStubs]
from beartype import beartype
from jaxtyping import Float, jaxtyped

Point2: TypeAlias = Float[np.ndarray, "2"]
BezierSegment: TypeAlias = tuple[Point2, Point2, Point2, Point2]


@jaxtyped(typechecker=beartype)
def catmull_rom_to_beziers(
    points: Float[np.ndarray, "M 2"],
    alpha: float = 0.5,
) -> list[BezierSegment]:
    """
    Convert a polyline to a sequence of cubic Bezier segments approximating Catmull-Rom spline.
    points: (M,2)
    Returns list of (p0, c1, c2, p3) per segment in world coords.
    """
    P = points
    M = P.shape[0]
    if M < 4:
        raise ValueError("Need at least 4 points for Catmull-Rom conversion.")

    segs: list[BezierSegment] = []
    for i in range(1, M - 2):
        p0 = P[i]
        p1 = P[i + 1]
        pm1 = P[i - 1]
        p2 = P[i + 2]

        # Catmull-Rom to Bezier control points
        c1 = p0 + (p1 - pm1) / 6.0
        c2 = p1 - (p2 - p0) / 6.0
        segs.append((p0, c1, c2, p1))
    return segs


@jaxtyped(typechecker=beartype)
def export_curves_svg(
    out_path: str,
    curves: Float[np.ndarray, "C M 2"],
    stroke: str = "#000000",
    stroke_width: float | str = 2.0,
    fill: str = "none",
    viewbox: tuple[float, float, float, float] | None = None,
    canvas_size: tuple[float, float] | tuple[str, str] | None = None,
    reference_shape: Float[np.ndarray, "N 2"] | None = None,
    reference_stroke: str = "#777777",
    reference_stroke_width: float | str = 1.0,
    reference_fill: str = "none",
    reference_opacity: float = 0.5,
    reference_dasharray: str | None = "4,4",
) -> None:
    """
    curves: (C,M,2) sampled points per curve in world coords
    reference_shape: (N,2) optional outline to draw for context
    """
    C, M, _ = curves.shape

    # Determine viewBox from data if not provided
    if viewbox is None:
        allp = curves.reshape(-1, 2)
        if reference_shape is not None:
            allp = np.vstack([allp, reference_shape])
        minx, miny = allp.min(axis=0)
        maxx, maxy = allp.max(axis=0)
        pad = 10.0
        viewbox = (
            float(minx - pad),
            float(miny - pad),
            float((maxx - minx) + 2 * pad),
            float((maxy - miny) + 2 * pad),
        )

    if canvas_size is None:
        dwg = svgwrite.Drawing(out_path, profile="tiny")
    else:
        dwg = svgwrite.Drawing(out_path, profile="tiny", size=canvas_size)
    viewbox_str = f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}"
    dwg.attribs["viewBox"] = viewbox_str

    def to_point_list(points: np.ndarray) -> list[tuple[float, float]]:
        return [(float(p[0]), float(p[1])) for p in points]

    if reference_shape is not None:
        ref_kwargs: dict[str, object] = {
            "stroke": reference_stroke,
            "fill": reference_fill,
            "stroke_width": reference_stroke_width,
            "opacity": reference_opacity,
        }
        if reference_dasharray is not None:
            ref_kwargs["stroke_dasharray"] = reference_dasharray
        dwg.add(
            dwg.polygon(
                points=to_point_list(reference_shape),
                **ref_kwargs,
            )
        )

    for i in range(C):
        pts = curves[i]
        # If too short, export as polyline
        if M < 6:
            dwg.add(
                dwg.polyline(
                    points=to_point_list(pts),
                    stroke=stroke,
                    fill=fill,
                    stroke_width=stroke_width,
                )
            )
            continue

        segs = catmull_rom_to_beziers(pts)
        # Build path string
        p = segs[0][0]
        d = [f"M {p[0]:.3f},{p[1]:.3f}"]
        for _p0, c1, c2, p3 in segs:
            d.append(
                f"C {c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {p3[0]:.3f},{p3[1]:.3f}"
            )
        path = dwg.path(
            d=" ".join(d),
            stroke=stroke,
            fill=fill,
            stroke_width=stroke_width,
        )
        dwg.add(path)

    dwg.save()
