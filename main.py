from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import trimesh

from autorigami.acap_integration import acap_submodule_status, compute_toolpath_stats
from autorigami.mesh_io import (
    build_parser,
    resolve_axis,
    save_overlay_obj,
    save_polyline_obj,
    timestamped_output_dir,
)
from autorigami.global_spiral import generate_global_spiral_candidates, to_polyline
from autorigami.parametrization import (
    Polyline,
    cubic_bspline_to_piecewise_bezier,
    fit_parametric_bspline_from_polyline,
    sample_cubic_bezier_chain,
    sample_parametric_bspline,
)
from autorigami.qc_plots import export_selected_candidate_qc_plots
from autorigami._native import (
    piecewise_hermite_generator,
    validate_piecewise_curve_curvature,
    validate_polyline_nonlocal_distance,
)


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


def polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))


def generate_initial_tangent_candidates() -> list[tuple[str, float]]:
    return [
        ("heading_0", 0.0),
        ("heading_p45", np.deg2rad(45.0)),
        ("heading_n45", np.deg2rad(-45.0)),
        ("heading_p90", np.deg2rad(90.0)),
        ("heading_n90", np.deg2rad(-90.0)),
        ("heading_180", np.deg2rad(180.0)),
    ]


def run_single_candidate(
    *,
    candidate_name: str,
    candidate_dir: Path,
    input_path: Path,
    mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    faces: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    axis_source: str,
    spacing_world: float,
    nonlocal_window_world: float,
    max_curvature: float,
    curvature_tolerance: float,
    extension_step_world: float,
    generator_rounds: int,
    use_single_seed: bool,
    initial_heading_angle_rad: float,
    fit_validation_samples: int,
    nonlocal_arc_window_nm: float,
    spacing_nm: float,
    args,
) -> dict[str, object]:
    piecewise_hermite, generator_run_data = piecewise_hermite_generator(
        vertices,
        faces,
        np.asarray(axis_origin, dtype=np.float64),
        np.asarray(axis_direction, dtype=np.float64),
        spacing_world,
        nonlocal_window_world,
        max_curvature,
        curvature_tolerance,
        extension_step_world,
        generator_rounds,
        use_single_seed,
        initial_heading_angle_rad,
    )

    curvature_validation_native_hermite = validate_piecewise_curve_curvature(
        piecewise_hermite=piecewise_hermite,
        max_curvature=max_curvature,
        curvature_tolerance=curvature_tolerance,
    )

    raw_control_points = np.asarray(piecewise_hermite.points, dtype=np.float64)
    raw_control_polyline = Polyline(points=raw_control_points)
    raw_curvature_max = estimate_polyline_curvature_max(raw_control_points)
    raw_nonlocal_result = validate_polyline_nonlocal_distance(
        points=raw_control_points,
        minimum_separation=spacing_world,
        nonlocal_window=nonlocal_window_world,
        stop_on_first_violation=False,
    )
    raw_nonlocal_violations = int(raw_nonlocal_result["violation_count"])
    raw_nonlocal_min_distance = float(raw_nonlocal_result["minimum_checked_distance"])

    smoothing_schedule = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
    chosen_smoothing: float | None = None
    chosen_splines = None
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
            sampled_fit_curvature_max = curvature_candidate
            sampled_fit_nonlocal_violations = violations_candidate
            sampled_fit_nonlocal_min_distance = min_distance_candidate
            break

    if chosen_splines is None:
        raise RuntimeError(
            f"Candidate {candidate_name}: no SciPy spline fit passed sampled validation"
        )

    cubic_bezier = cubic_bspline_to_piecewise_bezier(chosen_splines)
    polyline = sample_cubic_bezier_chain(cubic_bezier, num_samples=10000)

    source_mesh_copy_path = candidate_dir / input_path.name
    shutil.copy2(input_path, source_mesh_copy_path)
    polyline_obj_raw_path = candidate_dir / "spiral_polyline_raw.obj"
    save_polyline_obj(polyline, polyline_obj_raw_path)
    overlay_obj_path = candidate_dir / "spiral_overlay.obj"
    save_overlay_obj(mesh=mesh, polyline=polyline, output_obj_path=overlay_obj_path)

    run_info = {
        "candidate_name": candidate_name,
        "input_mesh": str(input_path),
        "generator": args.generator,
        "turns": args.turns,
        "samples": args.samples,
        "axis_mode": args.axis_mode,
        "axis": args.axis,
        "axis_source": axis_source,
        "resolved_axis_vector": axis_direction.tolist(),
        "resolved_axis_origin": axis_origin.tolist(),
        "initialization": {
            "use_single_seed": use_single_seed,
            "initial_heading_angle_rad": initial_heading_angle_rad,
        },
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
            "self_avoidance_min_distance_nm": spacing_nm,
            "self_avoidance_nonlocal_window_nm": nonlocal_arc_window_nm,
        },
        "external_toolpath_stats": {
            "acap": compute_toolpath_stats(
                points=raw_control_points,
                minimum_separation_world=spacing_world,
                nonlocal_window_world=nonlocal_window_world,
                world_to_nm=args.world_to_nm,
                nonlocal_violation_count=raw_nonlocal_violations,
                minimum_checked_distance_world=raw_nonlocal_min_distance,
            )
        },
    }
    run_info |= generator_run_data

    metadata_path = candidate_dir / "run_info.json"
    metadata_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    return {
        "candidate_name": candidate_name,
        "candidate_dir": str(candidate_dir),
        "run_info_path": str(metadata_path),
        "score_length_world": polyline_length(raw_control_points),
        "score_coverage_like_point_count": int(generator_run_data["cpp_point_count"]),
        "raw_control_nonlocal_violations": raw_nonlocal_violations,
        "fit_sample_nonlocal_violations": sampled_fit_nonlocal_violations,
        "fit_sample_curvature_max_world": sampled_fit_curvature_max,
        "max_curvature_threshold_world": max_curvature + curvature_tolerance,
    }


def is_candidate_feasible(summary: dict[str, object]) -> bool:
    if summary.get("status") != "ok":
        return False
    return (
        int(summary["raw_control_nonlocal_violations"]) == 0
        and int(summary["fit_sample_nonlocal_violations"]) == 0
        and float(summary["fit_sample_curvature_max_world"])
        <= float(summary["max_curvature_threshold_world"])
    )


def run_global_spiral_pipeline(
    *,
    input_path: Path,
    mesh: trimesh.Trimesh,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    axis_source: str,
    spacing_world: float,
    nonlocal_window_world: float,
    max_curvature: float,
    curvature_tolerance: float,
    nonlocal_arc_window_nm: float,
    spacing_nm: float,
    candidates_root: Path,
    args,
) -> list[dict[str, object]]:
    candidates = generate_global_spiral_candidates(
        mesh,
        axis_origin=np.asarray(axis_origin, dtype=np.float64),
        axis_direction=np.asarray(axis_direction, dtype=np.float64),
        spacing_world=spacing_world,
        nonlocal_window_world=nonlocal_window_world,
        samples=args.validation_samples,
    )
    summaries: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        candidate_dir = candidates_root / f"{index:03d}_{candidate.name}"
        candidate_dir.mkdir(parents=True, exist_ok=False)
        polyline = to_polyline(candidate)
        curvature_max = estimate_polyline_curvature_max(candidate.points)

        source_mesh_copy_path = candidate_dir / input_path.name
        shutil.copy2(input_path, source_mesh_copy_path)
        polyline_obj_raw_path = candidate_dir / "spiral_polyline_raw.obj"
        save_polyline_obj(polyline, polyline_obj_raw_path)
        overlay_obj_path = candidate_dir / "spiral_overlay.obj"
        save_overlay_obj(mesh=mesh, polyline=polyline, output_obj_path=overlay_obj_path)

        run_info = {
            "candidate_name": candidate.name,
            "input_mesh": str(input_path),
            "generator": args.generator,
            "axis_mode": args.axis_mode,
            "axis": args.axis,
            "axis_source": f"{axis_source}+global_spiral",
            "resolved_axis_vector": candidate.axis_direction.tolist(),
            "resolved_axis_origin": candidate.axis_origin.tolist(),
            "source_mesh_copy": str(source_mesh_copy_path),
            "polyline_obj_raw": str(polyline_obj_raw_path),
            "overlay_obj": str(overlay_obj_path),
            "global_spiral": {
                "turns": candidate.turns,
                "phase": candidate.phase,
                "axial_margin": candidate.axial_margin,
                "axis_direction": candidate.axis_direction.tolist(),
                "axis_origin": candidate.axis_origin.tolist(),
            },
            "validation": {
                "raw_control_nonlocal_violations": candidate.nonlocal_violations,
                "raw_control_nonlocal_min_distance_world": candidate.minimum_distance_world,
                "fit_sample_curvature_max_world": curvature_max,
                "fit_sample_nonlocal_violations": candidate.nonlocal_violations,
            },
            "generator_constraints": {
                "self_avoidance_min_distance_nm": spacing_nm,
                "self_avoidance_nonlocal_window_nm": nonlocal_arc_window_nm,
            },
            "external_toolpath_stats": {
                "acap": compute_toolpath_stats(
                    points=candidate.points,
                    minimum_separation_world=spacing_world,
                    nonlocal_window_world=nonlocal_window_world,
                    world_to_nm=args.world_to_nm,
                    nonlocal_violation_count=candidate.nonlocal_violations,
                    minimum_checked_distance_world=candidate.minimum_distance_world,
                )
            },
        }
        metadata_path = candidate_dir / "run_info.json"
        metadata_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

        summaries.append(
            {
                "candidate_name": candidate.name,
                "candidate_dir": str(candidate_dir),
                "run_info_path": str(metadata_path),
                "score_length_world": candidate.length_world,
                "score_coverage_like_point_count": int(candidate.points.shape[0]),
                "raw_control_nonlocal_violations": candidate.nonlocal_violations,
                "fit_sample_nonlocal_violations": candidate.nonlocal_violations,
                "fit_sample_curvature_max_world": curvature_max,
                "max_curvature_threshold_world": max_curvature + curvature_tolerance,
                "global_spiral_turns": candidate.turns,
                "global_spiral_min_distance_world": candidate.minimum_distance_world,
                "status": "ok",
            }
        )
    return summaries


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

    if args.generator not in {"piecewise_hermite", "global_spiral"}:
        raise ValueError(f"Unsupported generator: {args.generator}")

    max_curvature = args.world_to_nm / args.min_curvature_radius_nm
    curvature_tolerance = args.world_to_nm * 0.1 / (args.min_curvature_radius_nm**2)
    spacing_world = args.spacing_nm / args.world_to_nm
    nonlocal_window_world = nonlocal_arc_window_nm / args.world_to_nm
    extension_step_world = args.generator_step_nm / args.world_to_nm
    fit_validation_samples = min(args.validation_samples, 3000)

    parent_output_dir = timestamped_output_dir(args.output_root)
    candidates_root = parent_output_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)

    if args.generator == "global_spiral":
        candidate_summaries = run_global_spiral_pipeline(
            input_path=input_path,
            mesh=mesh,
            axis_origin=np.asarray(axis_origin, dtype=np.float64),
            axis_direction=np.asarray(axis_direction, dtype=np.float64),
            axis_source=axis_source,
            spacing_world=spacing_world,
            nonlocal_window_world=nonlocal_window_world,
            max_curvature=max_curvature,
            curvature_tolerance=curvature_tolerance,
            nonlocal_arc_window_nm=nonlocal_arc_window_nm,
            spacing_nm=args.spacing_nm,
            candidates_root=candidates_root,
            args=args,
        )
    else:
        candidate_headings = generate_initial_tangent_candidates()
        candidate_summaries: list[dict[str, object]] = []

        for index, (name, heading_angle_rad) in enumerate(candidate_headings):
            candidate_dir = candidates_root / f"{index:02d}_{name}"
            candidate_dir.mkdir(parents=True, exist_ok=False)
            try:
                summary = run_single_candidate(
                    candidate_name=name,
                    candidate_dir=candidate_dir,
                    input_path=input_path,
                    mesh=mesh,
                    vertices=vertices,
                    faces=faces,
                    axis_origin=np.asarray(axis_origin, dtype=np.float64),
                    axis_direction=np.asarray(axis_direction, dtype=np.float64),
                    axis_source=f"{axis_source}+tangent_sweep:{name}",
                    spacing_world=spacing_world,
                    nonlocal_window_world=nonlocal_window_world,
                    max_curvature=max_curvature,
                    curvature_tolerance=curvature_tolerance,
                    extension_step_world=extension_step_world,
                    generator_rounds=args.generator_rounds,
                    use_single_seed=True,
                    initial_heading_angle_rad=heading_angle_rad,
                    fit_validation_samples=fit_validation_samples,
                    nonlocal_arc_window_nm=nonlocal_arc_window_nm,
                    spacing_nm=args.spacing_nm,
                    args=args,
                )
                summary["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                summary = {
                    "candidate_name": name,
                    "candidate_dir": str(candidate_dir),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            candidate_summaries.append(summary)

    successful_candidates = [c for c in candidate_summaries if c.get("status") == "ok"]
    if not successful_candidates:
        raise RuntimeError("All initialization candidates failed; see selection_summary.json for details")

    feasible_candidates = [c for c in successful_candidates if is_candidate_feasible(c)]
    ranked_pool = feasible_candidates if feasible_candidates else successful_candidates
    ranked_pool.sort(
        key=lambda item: (
            float(item["score_length_world"]),
            int(item["score_coverage_like_point_count"]),
        ),
        reverse=True,
    )
    selected = ranked_pool[0]

    selected_dir = parent_output_dir / "selected_candidate"
    shutil.copytree(Path(str(selected["candidate_dir"])), selected_dir)
    qc_plot_summary = export_selected_candidate_qc_plots(selected_dir=selected_dir)

    selection_summary = {
        "selected_candidate": selected,
        "feasible_candidate_count": len(feasible_candidates),
        "total_candidate_count": len(candidate_summaries),
        "all_candidates": candidate_summaries,
        "qc_exports": {
            "selected_candidate_separation_histogram": qc_plot_summary,
        },
        "external_integrations": {
            "acap": acap_submodule_status(Path(__file__).resolve().parent),
        },
    }
    selection_summary_path = parent_output_dir / "selection_summary.json"
    selection_summary_path.write_text(
        json.dumps(selection_summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Output parent directory: {parent_output_dir}")
    print(f"Candidates directory: {candidates_root}")
    print(f"Selected candidate directory: {selected_dir}")
    print(f"Selected candidate separation histogram: {qc_plot_summary['output_png']}")
    print(f"Selection summary: {selection_summary_path}")


if __name__ == "__main__":
    main()
