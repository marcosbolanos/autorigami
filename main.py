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
    piecewise_hermite_to_bezier,
    sample_cubic_bezier_chain,
)
from autorigami._native import (
    piecewise_hermite_generator,
    validate_piecewise_curve_curvature,
)


def main() -> None:
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
        piecewise_hermite, generator_run_data = piecewise_hermite_generator(
            vertices, faces, axis_origin_array, axis_direction_array
        )
    else:  # we have room to add other generators here
        raise ValueError(f"Unsupported generator: {args.generator}")

    # Fast cpp validation function which handles piecewise hermite curves
    curvature_validation = validate_piecewise_curve_curvature(
        piecewise_hermite=piecewise_hermite,
        max_curvature=args.world_to_nm / args.min_curvature_radius_nm,
        curvature_tolerance=args.world_to_nm * 0.1 / (args.min_curvature_radius_nm**2),
    )

    # exact conversion to bezier for our outputs, and sampled polyline for viz
    cubic_bezier = piecewise_hermite_to_bezier(piecewise_hermite)
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
            "curvature_valid": curvature_validation,
        },
    }
    run_info |= generator_run_data
    metadata_path = output_dir / "run_info.json"
    metadata_path.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")

    print(f"Output directory: {output_dir}")
    print(f"Source mesh copy: {source_mesh_copy_path}")
    print(f"Raw polyline OBJ: {polyline_obj_raw_path}")
    print(f"Overlay OBJ: {overlay_obj_path}")
    print(f"Curvature validation: {'pass' if curvature_validation else 'fail'}")
    print(f"Resolved axis vector: {axis_direction.tolist()} ({axis_source})")
    print(f"Resolved axis origin: {axis_origin.tolist()}")
    print(f"Run info: {metadata_path}")


if __name__ == "__main__":
    main()
