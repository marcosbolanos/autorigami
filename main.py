from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

from autorigami.mesh_io import (
    build_parser,
    resolve_axis,
    save_overlay_obj,
    save_polyline_obj,
    timestamped_output_dir,
)
from autorigami.parametrization import (
    Polyline,
    cubic_bspline_to_piecewise_bezier,
    fit_parametric_bspline_from_polyline,
    sample_cubic_bezier_chain,
    sample_parametric_bspline,
)
from autorigami._native import (
    piecewise_hermite_generator,
    validate_polyline_nonlocal_distance,
    validate_piecewise_curve_curvature,
)


def main() -> None:
    nonlocal_arc_window_nm = 4.1
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() not in {".obj", ".stl"}:
        raise ValueError("Input mesh must be .obj or .stl")
    mesh = trimesh.load_mesh(input_path, force="mesh")
    axis_direction, axis_origin, axis_source = resolve_axis(mesh, args)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    axis_origin_array = np.asarray(axis_origin, dtype=np.float64)
    axis_direction_array = np.asarray(axis_direction, dtype=np.float64)

    if args.generator == "piecewise_hermite":
        max_curvature = args.world_to_nm / args.min_curvature_radius_nm
        curvature_tolerance = args.world_to_nm * 0.1 / (args.min_curvature_radius_nm**2)
        spacing_world = args.spacing_nm / args.world_to_nm
        nonlocal_window_world = nonlocal_arc_window_nm / args.world_to_nm
        extension_step_world = args.generator_step_nm / args.world_to_nm
        piecewise_hermite, generator_run_data = piecewise_hermite_generator(
            vertices,
            faces,
            axis_origin_array,
            axis_direction_array,
            spacing_world,
            nonlocal_window_world,
            max_curvature,
            curvature_tolerance,
            extension_step_world,
            args.generator_rounds,
        )
    else:  # we have room to add other generators here
        raise ValueError(f"Unsupported generator: {args.generator}")

    # Step 1: validate raw optimization output
    curvature_validation_native_hermite = validate_piecewise_curve_curvature(
        piecewise_hermite=piecewise_hermite,
        max_curvature=max_curvature,
        curvature_tolerance=curvature_tolerance,
    )
    raw_control_points = np.asarray(piecewise_hermite.points, dtype=np.float64)
    raw_control_polyline = Polyline(points=raw_control_points)

    def estimate_polyline_curvature_max(points: np.ndarray) -> float:
        if points.shape[0] < 3:
            return 0.0
        edges = points[1:] - points[:-1]
        lengths = np.linalg.norm(edges, axis=1)
        tangents = edges / np.maximum(lengths[:, None], 1e-12)
        delta_tangent = tangents[1:] - tangents[:-1]
        ds = 0.5 * (lengths[1:] + lengths[:-1])
        curvature = np.linalg.norm(delta_tangent, axis=1) / np.maximum(ds, 1e-12)
        return float(np.max(curvature))

    raw_curvature_max = estimate_polyline_curvature_max(raw_control_points)
    raw_nonlocal_result = validate_polyline_nonlocal_distance(
        points=raw_control_points,
        minimum_separation=spacing_world,
        nonlocal_window=nonlocal_window_world,
        stop_on_first_violation=False,
    )
    raw_nonlocal_violations = int(raw_nonlocal_result["violation_count"])
    raw_nonlocal_min_distance = float(raw_nonlocal_result["minimum_checked_distance"])

    # Step 2 + 3: fit parametric spline, sample, and validate sampled geometry
    smoothing_schedule = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    fit_validation_samples = min(args.validation_samples, 3000)
    chosen_smoothing: float | None = None
    chosen_splines = None
    sampled_fit_polyline: Polyline | None = None
    sampled_fit_curvature_max = float("inf")
    sampled_fit_nonlocal_violations = -1
    sampled_fit_nonlocal_min_distance = float("inf")

    for smoothing in smoothing_schedule:
        splines = fit_parametric_bspline_from_polyline(
            raw_control_polyline, smoothing=smoothing, degree=3
        )
        sampled_candidate = sample_parametric_bspline(
            splines, num_samples=fit_validation_samples
        )
        curvature_candidate = estimate_polyline_curvature_max(sampled_candidate.points)
        nonlocal_candidate_result = validate_polyline_nonlocal_distance(
            points=sampled_candidate.points,
            minimum_separation=spacing_world,
            nonlocal_window=nonlocal_window_world,
            stop_on_first_violation=True,
        )
        violations_candidate = int(nonlocal_candidate_result["violation_count"])
        min_distance_candidate = float(nonlocal_candidate_result["minimum_checked_distance"])
        if (
            curvature_candidate <= (max_curvature + curvature_tolerance)
            and violations_candidate == 0
        ):
            chosen_smoothing = smoothing
            chosen_splines = splines
            sampled_fit_polyline = sampled_candidate
            sampled_fit_curvature_max = curvature_candidate
            sampled_fit_nonlocal_violations = violations_candidate
            sampled_fit_nonlocal_min_distance = min_distance_candidate
            break

    if chosen_splines is None or sampled_fit_polyline is None:
        raise RuntimeError(
            "No SciPy parametric spline fit passed sampled curvature and nonlocal distance validation"
        )

    # Step 4: exact conversion from fitted cubic B-spline to Bezier
    cubic_bezier = cubic_bspline_to_piecewise_bezier(chosen_splines)
    polyline = sample_cubic_bezier_chain(cubic_bezier, num_samples=10000)

    output_dir = timestamped_output_dir(args.output_root)
    source_mesh_copy_path = output_dir / input_path.name
    shutil.copy2(input_path, source_mesh_copy_path)
    polyline_obj_raw_path = output_dir / "spiral_polyline_raw.obj"
    save_polyline_obj(polyline, polyline_obj_raw_path)
    overlay_obj_path = output_dir / "spiral_overlay.obj"
    save_overlay_obj(mesh=mesh, polyline=polyline, output_obj_path=overlay_obj_path)

    run_info = {
        "input_mesh": str(input_path),
        "generator": args.generator,
        "turns": args.turns,
        "samples": args.samples,
        "axis_mode": args.axis_mode,
        "axis": args.axis,
        "axis_source": axis_source,
        "resolved_axis_vector": axis_direction.tolist(),
        "resolved_axis_origin": axis_origin.tolist(),
        "source_mesh_copy": str(source_mesh_copy_path),
        "polyline_obj_raw": str(polyline_obj_raw_path),
        "overlay_obj": str(overlay_obj_path),
        "validation": {
            "curvature_valid_hermite_native": curvature_validation_native_hermite,
            "raw_control_curvature_max_world": raw_curvature_max,
            "raw_control_nonlocal_violations": raw_nonlocal_violations,
            "raw_control_nonlocal_min_distance_world": raw_nonlocal_min_distance,
            "fit_smoothing": chosen_smoothing,
            "fit_sample_curvature_max_world": sampled_fit_curvature_max,
            "fit_sample_nonlocal_violations": sampled_fit_nonlocal_violations,
            "fit_sample_nonlocal_min_distance_world": sampled_fit_nonlocal_min_distance,
            "bezier_fit_source": "scipy_parametric_bspline_exact_to_bezier",
        },
        "generator_constraints": {
            "self_avoidance_min_distance_nm": args.spacing_nm,
            "self_avoidance_nonlocal_window_nm": nonlocal_arc_window_nm,
        },
    }
    run_info |= generator_run_data
    metadata_path = output_dir / "run_info.json"
    metadata_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Source mesh copy: {source_mesh_copy_path}")
    print(f"Raw polyline OBJ: {polyline_obj_raw_path}")
    print(f"Overlay OBJ: {overlay_obj_path}")
    print(
        "Curvature validation (native Hermite): "
        f"{'pass' if curvature_validation_native_hermite else 'fail'}"
    )
    print(f"Resolved axis vector: {axis_direction.tolist()} ({axis_source})")
    print(f"Resolved axis origin: {axis_origin.tolist()}")
    print(f"Run info: {metadata_path}")


if __name__ == "__main__":
    main()
