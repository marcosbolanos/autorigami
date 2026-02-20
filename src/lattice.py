from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from beartype import beartype
from jaxtyping import Float, jaxtyped
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


@jaxtyped(typechecker=beartype)
def honeycomb_lattice_points(
    V: Float[np.ndarray, "N 2"],
    neighbor_dist: float,
    *,
    origin: Float[np.ndarray, "2"] | None = None,
    pad: float = 0.0,
) -> Float[np.ndarray, "M 2"]:
    """Generate honeycomb (graphene) lattice vertices inside a polygon.

    The output points form the 2D honeycomb lattice with 3 nearest neighbors per vertex.

    Parameters
    - V: (N,2) polygon vertices in *SVG/user* units (closed or open).
    - neighbor_dist: nearest-neighbor distance in the same units as V.
      In the SVG pipeline this should be `neighbor_dist_nm / nm_per_unit`.
    - origin: optional lattice origin in the same units as V.
      If None, defaults to the polygon bbox (minx, miny), making the lattice translation-
      equivariant with the input polygon.
    - pad: extra margin (units) added around the polygon bbox when generating candidates.

    Returns
    - P: (M,2) float32 lattice vertices inside/on the polygon.
    """

    if not np.isfinite(neighbor_dist) or neighbor_dist <= 0:
        raise ValueError("neighbor_dist must be a finite positive number")
    if not np.isfinite(pad) or pad < 0:
        raise ValueError("pad must be finite and >= 0")

    V_arr = np.asarray(V, dtype=np.float32)
    if V_arr.ndim != 2 or V_arr.shape[1] != 2:
        raise ValueError("V must have shape (N,2)")
    if V_arr.shape[0] < 3:
        raise ValueError("V must contain at least 3 vertices")
    if not np.isfinite(V_arr).all():
        raise ValueError("V contains non-finite coordinates")

    poly = Polygon(V_arr)
    if not poly.is_valid:
        fixed = poly.buffer(0)
        if fixed.is_empty:
            raise ValueError("Polygon is empty after fixing")
        fixed_any = cast(Any, fixed)
        if getattr(fixed_any, "geom_type", None) == "MultiPolygon":
            # Shouldn't happen for the single-outline SVGs we expect, but be defensive.
            poly = max(fixed_any.geoms, key=lambda g: g.area)
        else:
            poly = fixed_any
    if poly.is_empty:
        raise ValueError("Polygon is empty")

    minx, miny, maxx, maxy = poly.bounds
    if origin is None:
        origin_arr = np.array([minx, miny], dtype=np.float32)
    else:
        origin_arr = np.asarray(origin, dtype=np.float32)
        if origin_arr.shape != (2,):
            raise ValueError("origin must have shape (2,)")
        if not np.isfinite(origin_arr).all():
            raise ValueError("origin contains non-finite coordinates")

    d = float(neighbor_dist)
    dx = math.sqrt(3.0) * d
    dy = 1.5 * d
    if dx <= 0 or dy <= 0:
        raise ValueError("neighbor_dist must be positive")

    # Honeycomb lattice as 2-point basis on a triangular Bravais lattice.
    # Bravais vectors:
    #   b1 = (dx, 0)
    #   b2 = (dx/2, dy)
    # Basis:
    #   A = (0, 0)
    #   B = (dx/2, d/2)
    # This yields nearest-neighbor distance exactly d with 3 neighbors per vertex.
    basis_a = (0.0, 0.0)
    basis_b = (dx / 2.0, d / 2.0)

    # Candidate generation ranges over bbox (with padding) in a conservative way.
    x_min = float(minx - pad)
    x_max = float(maxx + pad)
    y_min = float(miny - pad)
    y_max = float(maxy + pad)

    oy = float(origin_arr[1])
    n_min = math.floor((y_min - oy - max(basis_a[1], basis_b[1])) / dy) - 2
    n_max = math.ceil((y_max - oy - min(basis_a[1], basis_b[1])) / dy) + 2

    ox = float(origin_arr[0])
    pts: list[tuple[float, float]] = []
    for n in range(n_min, n_max + 1):
        y_a = oy + n * dy + basis_a[1]
        y_b = oy + n * dy + basis_b[1]

        x_shift = ox + n * (dx / 2.0)

        m_min_a = math.floor((x_min - x_shift - basis_a[0]) / dx) - 2
        m_max_a = math.ceil((x_max - x_shift - basis_a[0]) / dx) + 2
        for m in range(m_min_a, m_max_a + 1):
            pts.append((x_shift + m * dx + basis_a[0], y_a))

        m_min_b = math.floor((x_min - x_shift - basis_b[0]) / dx) - 2
        m_max_b = math.ceil((x_max - x_shift - basis_b[0]) / dx) + 2
        for m in range(m_min_b, m_max_b + 1):
            pts.append((x_shift + m * dx + basis_b[0], y_b))

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)

    P = np.asarray(pts, dtype=np.float32)

    prepped = prep(poly)
    covers_fn: Callable[[Point], bool] | None = getattr(prepped, "covers", None)
    if covers_fn is None:
        # Older shapely prepared geometries may not implement covers().
        covers_fn = poly.covers
    covers_fn = cast(Callable[[Point], bool], covers_fn)
    keep = np.fromiter(
        (covers_fn(Point(float(x), float(y))) for x, y in cast(np.ndarray, P)),
        dtype=bool,
        count=P.shape[0],
    )
    P_in = P[keep]
    if P_in.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Deterministic ordering: sort by x then y.
    order = np.lexsort((P_in[:, 1], P_in[:, 0]))
    return P_in[order].astype(np.float32)


def _export_polygon_and_points_svg(
    out_path: str,
    *,
    V: np.ndarray,
    P: np.ndarray,
    viewbox: tuple[float, float, float, float],
    canvas_size: tuple[str, str] | None,
    point_radius: float,
    point_fill: str = "#d11",
    outline_stroke: str = "#111",
    outline_stroke_width: float | str = "1pt",
) -> None:
    import svgwrite  # type: ignore[reportMissingTypeStubs]

    dwg = (
        svgwrite.Drawing(out_path, profile="tiny")
        if canvas_size is None
        else svgwrite.Drawing(out_path, profile="tiny", size=canvas_size)
    )
    dwg.attribs["viewBox"] = f"{viewbox[0]} {viewbox[1]} {viewbox[2]} {viewbox[3]}"

    def to_point_list(points: np.ndarray) -> list[tuple[float, float]]:
        return [(float(p[0]), float(p[1])) for p in points]

    dwg.add(
        dwg.polygon(
            points=to_point_list(V),
            stroke=outline_stroke,
            fill="none",
            stroke_width=outline_stroke_width,
            opacity=0.9,
        )
    )

    g = dwg.g(id="lattice_points", fill=point_fill, stroke="none", opacity=0.85)
    for x, y in P:
        g.add(dwg.circle(center=(float(x), float(y)), r=float(point_radius)))
    dwg.add(g)
    dwg.save()


def _default_viewbox(V: np.ndarray, pad: float) -> tuple[float, float, float, float]:
    minx, miny = V.min(axis=0)
    maxx, maxy = V.max(axis=0)
    return (
        float(minx - pad),
        float(miny - pad),
        float((maxx - minx) + 2.0 * pad),
        float((maxy - miny) + 2.0 * pad),
    )


def _resolve_repo_path(rel_path: str) -> Path:
    # src/ is the package root; PROJECT_ROOT is one directory above it.
    project_root = Path(__file__).resolve().parent.parent
    return project_root / rel_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a honeycomb lattice preview SVG")
    ap.add_argument(
        "--input",
        type=str,
        default=str(_resolve_repo_path("data/raw/blob1.svg")),
        help="Input SVG path (default: data/raw/blob1.svg)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(_resolve_repo_path("data/processed/blob1_honeycomb_lattice.svg")),
        help="Output SVG path",
    )
    ap.add_argument(
        "--neighbor_nm",
        type=float,
        default=2.6,
        help="Nearest-neighbor spacing in nm (default: 2.6)",
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
        help="SVG flatten tolerance in nm (default: 2.0)",
    )
    ap.add_argument(
        "--dot_r_nm",
        type=float,
        default=0.25,
        help="Lattice point radius in nm for visualization (default: 0.25)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input SVG not found; skipping: {in_path}")
        return

    if args.neighbor_nm <= 0:
        raise ValueError("neighbor_nm must be positive")
    if args.scale_nm_per_cm <= 0:
        raise ValueError("scale_nm_per_cm must be positive")
    if args.flat_tol_nm <= 0:
        raise ValueError("flat_tol_nm must be positive")
    if args.dot_r_nm <= 0:
        raise ValueError("dot_r_nm must be positive")

    from src.curvepack.svg_io import (
        load_single_path_polygon,
        load_svg_canvas,
        load_svg_scale_nm,
    )

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

    neighbor_units = float(args.neighbor_nm) / float(nm_per_unit)
    dot_r_units = float(args.dot_r_nm) / float(nm_per_unit)
    P = honeycomb_lattice_points(V, neighbor_dist=neighbor_units)

    if viewbox is None:
        viewbox = _default_viewbox(V, pad=2.0 * neighbor_units)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _export_polygon_and_points_svg(
        str(out_path),
        V=V,
        P=P,
        viewbox=viewbox,
        canvas_size=canvas_size,
        point_radius=dot_r_units,
        point_fill="#d11",
        outline_stroke="#111",
        outline_stroke_width="1pt",
    )

    print(
        f"Saved: {out_path}  points={P.shape[0]}  "
        f"d_nm={float(args.neighbor_nm):.6g} d_units={neighbor_units:.6g}"
    )


if __name__ == "__main__":
    main()
