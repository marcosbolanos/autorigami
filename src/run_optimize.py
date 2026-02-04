from __future__ import annotations

import argparse
from typing import Protocol, cast

import numpy as np

from .curvepack.svg_io import load_single_path_polygon
from .curvepack.sdf import polygon_to_mask, mask_to_sdf, sample_interior_points
from .curvepack.bspline import make_basis_matrix
from .curvepack.optimize import sample_init_control_points, optimize_curves
from .curvepack.export_svg import export_curves_svg


class CliArgs(Protocol):
    input: str
    output: str
    flat_tol: float
    h: float
    Q: int
    C: int
    n_ctrl: int
    M: int
    steps: int
    lr: float
    seed: int
    tube_r: float
    delta: float
    Rmin: float
    w_inside: float
    w_sep: float
    w_curv: float
    w_fill: float
    tau_fill: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="Input SVG with one closed outline path"
    )
    ap.add_argument("--output", required=True, help="Output SVG for optimized curves")
    ap.add_argument(
        "--flat_tol",
        type=float,
        default=2.0,
        help="SVG flatten tolerance (world units)",
    )
    ap.add_argument(
        "--h", type=float, default=2.0, help="SDF grid spacing (world units)"
    )
    ap.add_argument(
        "--Q", type=int, default=4000, help="Number of interior samples for fill"
    )
    ap.add_argument("--C", type=int, default=6, help="Number of curves")
    ap.add_argument("--n_ctrl", type=int, default=10, help="Control points per curve")
    ap.add_argument(
        "--M", type=int, default=80, help="Samples per curve for constraints"
    )
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=0)

    # Geometry / constraints
    ap.add_argument(
        "--tube_r", type=float, default=6.0, help="Tube radius (world units)"
    )
    ap.add_argument(
        "--delta", type=float, default=2.0, help="Extra clearance between tubes"
    )
    ap.add_argument("--Rmin", type=float, default=30.0, help="Minimum bend radius")

    # Loss weights
    ap.add_argument("--w_inside", type=float, default=8.0)
    ap.add_argument("--w_sep", type=float, default=8.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_fill", type=float, default=3.0)
    ap.add_argument("--tau_fill", type=float, default=2.0)

    args = cast(CliArgs, ap.parse_args())

    rng = np.random.default_rng(args.seed)

    # 1) SVG -> polygon
    V = load_single_path_polygon(args.input, flat_tol=args.flat_tol)

    # 2) Polygon -> mask -> SDF
    mask, origin, h = polygon_to_mask(V, h=args.h, pad=20.0)
    sdf = mask_to_sdf(mask, h=h)

    # 3) Interior samples Y
    Y = sample_interior_points(mask, origin, h, Q=args.Q, rng=rng)

    # 4) B-spline basis
    B = make_basis_matrix(n_ctrl=args.n_ctrl, n_samples=args.M, degree=3)

    # 5) Init curves
    P0 = sample_init_control_points(Y, C=args.C, n_ctrl=args.n_ctrl, rng=rng)
    r = np.full((args.C,), args.tube_r, dtype=np.float32)

    # 6) Optimize
    P_opt, last_L = optimize_curves(
        P0,
        B,
        sdf,
        origin,
        h,
        Y,
        r,
        steps=args.steps,
        lr=args.lr,
        Rmin=args.Rmin,
        delta=args.delta,
        r_fill=args.tube_r,
        tau_fill=args.tau_fill,
        w_inside=args.w_inside,
        w_curv=args.w_curv,
        w_sep=args.w_sep,
        w_fill=args.w_fill,
        seed=args.seed,
    )

    # 7) Sample final curves and export SVG
    X_opt = np.einsum("mn,cnd->cmd", B, P_opt)  # (C,M,2)

    # Use viewBox derived from the original polygon for consistency
    minx, miny = V.min(axis=0)
    maxx, maxy = V.max(axis=0)
    pad = 20.0
    viewbox = (
        float(minx - pad),
        float(miny - pad),
        float(maxx - minx + 2 * pad),
        float(maxy - miny + 2 * pad),
    )

    export_curves_svg(
        args.output, X_opt, stroke="#111", stroke_width=2.5, viewbox=viewbox
    )
    print(f"Saved: {args.output}  final_loss={last_L:.6g}")


if __name__ == "__main__":
    main()
