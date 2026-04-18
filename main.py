from __future__ import annotations

import json
from pathlib import Path
import shutil

import trimesh

from autorigami.mesh_io import (
    build_parser,
    resolve_axis,
    save_overlay_obj,
    save_polyline_obj,
    timestamped_output_dir,
)
from autorigami.metrics import compute_polyline_metrics
from autorigami.parametrization import (
    polyline_to_cubic_bezier_chain,
    sample_cubic_bezier_chain,
)
from autorigami._native import validate_polyline_constraints
from autorigami.spiral_generation import generate_spiral_on_surface, generate_tight_spiral_ode


def main() -> None:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() not in {".obj", ".stl"}:
        raise ValueError("Input mesh must be .obj or .stl")
    mesh = trimesh.load_mesh(input_path, force="mesh")
    axis_direction, axis_origin, axis_source = resolve_axis(mesh, args)

    if args.generator == "simple_spiral":
        polyline = generate_spiral_on_surface(
            mesh=mesh,
            turns=args.turns,
            samples=args.samples,
            axis_direction=axis_direction,
            axis_origin=axis_origin,
        )
    elif args.generator == "ode":
        polyline = generate_tight_spiral_ode(
            mesh=mesh,
            samples=args.samples,
            axis_direction=axis_direction,
            axis_origin=axis_origin,
            world_to_nm=args.world_to_nm,
            target_spacing_nm=args.spacing_nm,
            min_curvature_radius_nm=args.min_curvature_radius_nm,
            repulsion_strength=args.repulsion_strength,
            repulsion_range_nm=args.repulsion_range_nm,
            repulsion_lag_points=args.repulsion_lag_points,
            tangential_speed_nm=args.tangential_speed_nm,
            step_size_nm=args.step_size_nm,
            min_progress_fraction=args.min_progress_fraction,
            bottom_clearance_nm=args.bottom_clearance_nm,
            top_clearance_nm=args.top_clearance_nm,
        )
    else:  # unsupported but we'll add room for other generators here
        raise ValueError(f"Unsupported generator: {args.generator}")

    bezier_chain = polyline_to_cubic_bezier_chain(polyline)
    bezier_validation_samples = sample_cubic_bezier_chain(
        bezier_chain,
        num_samples=args.validation_samples,
    )
    validation = validate_polyline_constraints(
        points=bezier_validation_samples,
        separation=args.spacing_nm / args.world_to_nm,
        max_curvature=args.world_to_nm / args.min_curvature_radius_nm,
    )
    metrics = compute_polyline_metrics(
        points=bezier_validation_samples,
        mesh=mesh,
        axis_direction=axis_direction,
        axis_origin=axis_origin,
        world_to_nm=args.world_to_nm,
        separation_nm=args.spacing_nm,
    )

    output_dir = timestamped_output_dir(args.output_root)
    source_mesh_copy_path = output_dir / input_path.name
    shutil.copy2(input_path, source_mesh_copy_path)
    polyline_obj_raw_path = output_dir / "spiral_polyline_raw.obj"
    save_polyline_obj(polyline, polyline_obj_raw_path)
    bezier_samples_obj_path = output_dir / "spiral_bezier_samples.obj"
    save_polyline_obj(bezier_validation_samples, bezier_samples_obj_path)
    overlay_obj_path = output_dir / "spiral_overlay.obj"
    save_overlay_obj(mesh=mesh, polyline=bezier_validation_samples, output_obj_path=overlay_obj_path)

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
        "num_polyline_points": int(polyline.shape[0]),
        "num_bezier_segments": int(bezier_chain.shape[0]),
        "num_validation_samples": int(bezier_validation_samples.shape[0]),
        "source_mesh_copy": str(source_mesh_copy_path),
        "polyline_obj_raw": str(polyline_obj_raw_path),
        "bezier_samples_obj": str(bezier_samples_obj_path),
        "overlay_obj": str(overlay_obj_path),
        "validation": {
            "separation_compliant": validation.separation.compliant_count,
            "separation_total": validation.separation.total_count,
            "curvature_compliant": validation.curvature.compliant_count,
            "curvature_total": validation.curvature.total_count,
        },
        "metrics": {
            "length_nm": metrics.length_nm,
            "length_world": metrics.length_world,
            "nearest_nonlocal_separation_nm": {
                "count": metrics.nearest_nonlocal_separation.count,
                "min": metrics.nearest_nonlocal_separation.min_nm,
                "mean": metrics.nearest_nonlocal_separation.mean_nm,
                "q25": metrics.nearest_nonlocal_separation.q25_nm,
                "q75": metrics.nearest_nonlocal_separation.q75_nm,
                "max": metrics.nearest_nonlocal_separation.max_nm,
            },
            "axis_coverage": {
                "polyline_min_nm": metrics.axis_coverage.polyline_min_nm,
                "polyline_max_nm": metrics.axis_coverage.polyline_max_nm,
                "mesh_min_nm": metrics.axis_coverage.mesh_min_nm,
                "mesh_max_nm": metrics.axis_coverage.mesh_max_nm,
                "span_nm": metrics.axis_coverage.span_nm,
                "mesh_span_nm": metrics.axis_coverage.mesh_span_nm,
                "span_ratio": metrics.axis_coverage.span_ratio,
                "start_ratio": metrics.axis_coverage.start_ratio,
                "end_ratio": metrics.axis_coverage.end_ratio,
            },
        },
    }
    if args.generator == "ode":
        run_info["ode_params"] = {
            "world_to_nm": args.world_to_nm,
            "spacing_nm": args.spacing_nm,
            "min_curvature_radius_nm": args.min_curvature_radius_nm,
            "repulsion_strength": args.repulsion_strength,
            "repulsion_range_nm": args.repulsion_range_nm,
            "repulsion_lag_points": args.repulsion_lag_points,
            "tangential_speed_nm": args.tangential_speed_nm,
            "step_size_nm": args.step_size_nm,
            "min_progress_fraction": args.min_progress_fraction,
            "bottom_clearance_nm": args.bottom_clearance_nm,
            "top_clearance_nm": args.top_clearance_nm,
        }
    metadata_path = output_dir / "run_info.json"
    metadata_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Source mesh copy: {source_mesh_copy_path}")
    print(f"Raw polyline OBJ: {polyline_obj_raw_path}")
    print(f"Bezier sample OBJ: {bezier_samples_obj_path}")
    print(f"Overlay OBJ: {overlay_obj_path}")
    print(
        "Validation separation compliance: "
        f"{validation.separation.compliant_count}/{validation.separation.total_count}"
    )
    print(
        "Validation curvature compliance: "
        f"{validation.curvature.compliant_count}/{validation.curvature.total_count}"
    )
    print(f"Polyline length: {metrics.length_nm:.2f} nm")
    print(
        "Nearest nonlocal separation (nm): "
        f"min={metrics.nearest_nonlocal_separation.min_nm:.3f}, "
        f"mean={metrics.nearest_nonlocal_separation.mean_nm:.3f}, "
        f"q25={metrics.nearest_nonlocal_separation.q25_nm:.3f}, "
        f"q75={metrics.nearest_nonlocal_separation.q75_nm:.3f}, "
        f"max={metrics.nearest_nonlocal_separation.max_nm:.3f}"
    )
    print(
        "Axis coverage: "
        f"{metrics.axis_coverage.span_ratio * 100.0:.1f}% span, "
        f"start={metrics.axis_coverage.start_ratio * 100.0:.1f}%, "
        f"end={metrics.axis_coverage.end_ratio * 100.0:.1f}%"
    )
    print(f"Resolved axis vector: {axis_direction.tolist()} ({axis_source})")
    print(f"Resolved axis origin: {axis_origin.tolist()}")
    print(f"Run info: {metadata_path}")


if __name__ == "__main__":
    main()
