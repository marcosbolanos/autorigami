from __future__ import annotations

import argparse
from typing import Protocol, cast

import numpy as np

from .curvepack.svg_io import (
    load_single_path_polygon,
    load_svg_canvas,
    load_svg_scale_nm,
)
from .curvepack.sdf import polygon_to_mask, mask_to_sdf, sample_interior_points
from .curvepack.bspline import make_basis_matrix
from .curvepack.optimize import sample_init_control_points, optimize_curves
from .curvepack.export_svg import export_curves_svg
from .utils import debug, debug_helpers


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
    scale_nm_per_cm: float
    checkpoint_every: int
    tube_r: float
    delta: float
    Rmin: float
    init_len: float
    init_jitter: float
    w_inside: float
    w_sep: float
    w_curv: float
    w_fill: float
    tau_fill: float
    verbose: bool


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
        help="SVG flatten tolerance (nm)",
    )
    ap.add_argument("--h", type=float, default=2.0, help="SDF grid spacing (nm)")
    ap.add_argument(
        "--Q", type=int, default=4000, help="Number of interior samples for fill"
    )
    ap.add_argument("--C", type=int, default=6, help="Number of curves")
    ap.add_argument("--n_ctrl", type=int, default=10, help="Control points per curve")
    ap.add_argument(
        "--M", type=int, default=80, help="Samples per curve for constraints"
    )
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable debug logs")
    ap.add_argument(
        "--scale_nm_per_cm",
        type=float,
        default=1.0,
        help="Real nm per SVG cm (default: 1cm -> 1nm)",
    )
    ap.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help="Checkpoint interval for NPZ/SVG (0 disables NPZ/SVG)",
    )

    # Geometry / constraints
    ap.add_argument("--tube_r", type=float, default=1.0, help="Tube radius (nm)")
    ap.add_argument(
        "--delta", type=float, default=0.6, help="Extra clearance between tubes (nm)"
    )
    ap.add_argument("--Rmin", type=float, default=6.0, help="Minimum bend radius (nm)")
    ap.add_argument(
        "--init_len",
        type=float,
        default=4.0,
        help="Initial segment length in tube radii",
    )
    ap.add_argument(
        "--init_jitter",
        type=float,
        default=0.0,
        help="Initial perpendicular jitter in tube radii",
    )

    # Loss weights
    ap.add_argument("--w_inside", type=float, default=8.0)
    ap.add_argument("--w_sep", type=float, default=8.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_fill", type=float, default=3.0)
    ap.add_argument("--tau_fill", type=float, default=2.0, help="Fill softness (nm)")

    args = cast(CliArgs, ap.parse_args())
    debug.set_verbose(args.verbose)

    rng = np.random.default_rng(args.seed)

    if args.scale_nm_per_cm <= 0:
        raise ValueError("scale_nm_per_cm must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")

    viewbox, canvas_size = load_svg_canvas(args.input)
    nm_per_unit = load_svg_scale_nm(args.input, nm_per_cm=args.scale_nm_per_cm)
    if nm_per_unit is None or nm_per_unit <= 0:
        nm_per_unit = 1.0
        if debug.is_verbose():
            debug.log("svg_scale: missing physical units; assuming 1 nm per SVG unit")
    else:
        debug.log(
            "svg_scale "
            f"nm_per_unit={nm_per_unit:.6g} nm_per_cm={args.scale_nm_per_cm:.6g}"
        )

    flat_tol_units = args.flat_tol / nm_per_unit
    h_units = args.h / nm_per_unit
    tube_r_units = args.tube_r / nm_per_unit
    delta_units = args.delta / nm_per_unit
    Rmin_units = args.Rmin / nm_per_unit
    tau_fill_units = args.tau_fill / nm_per_unit
    mask_pad_units = 20.0 / nm_per_unit

    # 1) SVG -> polygon
    V = load_single_path_polygon(args.input, flat_tol=flat_tol_units)
    debug_helpers.log_array("V", V)

    # 2) Polygon -> mask -> SDF
    mask, origin, h = polygon_to_mask(V, h=h_units, pad=mask_pad_units)
    debug.log(
        f"mask: shape={mask.shape} inside={int(mask.sum())} "
        f"origin=({origin[0]:.6g},{origin[1]:.6g}) h={h:.6g}"
    )
    sdf = mask_to_sdf(mask, h=h)
    debug_helpers.log_array("sdf", sdf)

    # 3) Interior samples Y
    Y = sample_interior_points(mask, origin, h, Q=args.Q, rng=rng)
    debug_helpers.log_array("Y", Y)

    # 4) B-spline basis
    B = make_basis_matrix(n_ctrl=args.n_ctrl, n_samples=args.M, degree=3)
    debug_helpers.log_array("B", B)

    # 5) Init curves
    init_len_units = args.init_len * tube_r_units
    init_jitter_units = args.init_jitter * tube_r_units
    P0 = sample_init_control_points(
        Y,
        C=args.C,
        n_ctrl=args.n_ctrl,
        rng=rng,
        init_len=init_len_units,
        init_jitter=init_jitter_units,
    )
    debug_helpers.log_array("P0", P0)
    r = np.full((args.C,), tube_r_units, dtype=np.float32)
    debug_helpers.log_array("r", r)

    # 6) Optimize
    if viewbox is None:
        minx, miny = V.min(axis=0)
        maxx, maxy = V.max(axis=0)
        viewbox = (
            float(minx),
            float(miny),
            float(maxx - minx),
            float(maxy - miny),
        )
    if canvas_size is None:
        canvas_size = (float(viewbox[2]), float(viewbox[3]))

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
        Rmin=Rmin_units,
        delta=delta_units,
        r_fill=tube_r_units,
        tau_fill=tau_fill_units,
        w_inside=args.w_inside,
        w_curv=args.w_curv,
        w_sep=args.w_sep,
        w_fill=args.w_fill,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_scale_nm_per_unit=nm_per_unit,
        checkpoint_scale_nm_per_cm=args.scale_nm_per_cm,
        checkpoint_viewbox=viewbox,
        checkpoint_canvas_size=canvas_size,
        checkpoint_stroke_width="1pt",
    )

    # 7) Sample final curves and export SVG
    X_opt = np.einsum("mn,cnd->cmd", B, P_opt)  # (C,M,2)

    export_curves_svg(
        args.output,
        X_opt,
        stroke="#111",
        stroke_width="1pt",
        viewbox=viewbox,
        canvas_size=canvas_size,
    )
    print(f"Saved: {args.output}  final_loss={last_L:.6g}")


if __name__ == "__main__":
    main()
