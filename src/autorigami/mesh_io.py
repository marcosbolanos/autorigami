from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a spiral polyline on a mesh surface.")
    parser.add_argument(
        "--input",
        default="assets/ellipsoid.obj",
        help="Input mesh path (.obj or .stl). Default: assets/ellipsoid.obj",
    )
    parser.add_argument(
        "--generator",
        default="simple_spiral",
        choices=["simple_spiral", "ode"],
        help="Spiral generation algorithm.",
    )
    parser.add_argument("--turns", type=float, default=10.0, help="Number of spiral turns.")
    parser.add_argument("--samples", type=int, default=1500, help="Number of polyline points.")
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=20000,
        help="Number of points sampled from Bezier chain for validation/export.",
    )
    parser.add_argument(
        "--world-to-nm",
        type=float,
        default=20.0,
        help="Scale factor from world units to nanometers (default fits ellipsoid scale).",
    )
    parser.add_argument(
        "--spacing-nm",
        type=float,
        default=2.6,
        help="Target spiral self-separation in nanometers.",
    )
    parser.add_argument(
        "--min-curvature-radius-nm",
        type=float,
        default=6.0,
        help="Minimum allowed local curvature radius in nanometers.",
    )
    parser.add_argument(
        "--repulsion-strength",
        type=float,
        default=2.0,
        help="Weight for self-repulsive potential in ODE generator.",
    )
    parser.add_argument(
        "--repulsion-range-nm",
        type=float,
        default=2.6,
        help="Repulsion interaction range in nanometers for ODE generator.",
    )
    parser.add_argument(
        "--repulsion-lag-points",
        type=int,
        default=8,
        help="Ignore latest N points in self-repulsion to avoid immediate self-push.",
    )
    parser.add_argument(
        "--tangential-speed-nm",
        type=float,
        default=10.0,
        help="Base tangential drive magnitude (nm per integration step-unit).",
    )
    parser.add_argument(
        "--step-size-nm",
        type=float,
        default=0.6,
        help="Integration step length in nanometers for ODE generator.",
    )
    parser.add_argument(
        "--axis",
        default="x",
        choices=["x", "y", "z"],
        help="Primary axis used in bbox mode.",
    )
    parser.add_argument(
        "--axis-vector",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Custom axis direction vector. Overrides --axis and --axis-mode.",
    )
    parser.add_argument(
        "--axis-origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Point on the axis. Default is mesh centroid.",
    )
    parser.add_argument(
        "--axis-mode",
        default="bbox",
        choices=["bbox", "inertia", "manual"],
        help="Axis resolution mode when --axis-vector is not provided.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output directory; a timestamp subfolder is created inside it.",
    )
    return parser


def timestamped_output_dir(output_root: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_polyline_csv(polyline: np.ndarray, output_path: Path) -> None:
    header = "x,y,z"
    np.savetxt(output_path, polyline, delimiter=",", header=header, comments="")


def save_polyline_obj(polyline: np.ndarray, output_path: Path) -> None:
    if polyline.shape[0] < 2:
        raise ValueError("Polyline must contain at least 2 points.")

    lines: list[str] = []
    for x, y, z in polyline:
        lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
    indices = " ".join(str(i) for i in range(1, polyline.shape[0] + 1))
    lines.append(f"l {indices}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_overlay_obj(mesh: trimesh.Trimesh, polyline: np.ndarray, output_obj_path: Path) -> None:
    if polyline.shape[0] < 2:
        raise ValueError("Polyline must contain at least 2 points.")

    output_mtl_path = output_obj_path.with_suffix(".mtl")
    output_mtl_path.write_text(
        "\n".join(
            [
                "newmtl mesh_white",
                "Ka 0.2 0.2 0.2",
                "Kd 0.95 0.95 0.95",
                "Ks 0.8 0.8 0.8",
                "Ns 200.0",
                "illum 2",
                "",
                "newmtl spiral_line",
                "Ka 0.0 0.0 0.0",
                "Kd 0.0 0.0 0.0",
                "Ks 0.0 0.0 0.0",
                "Ns 10.0",
                "illum 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals)

    lines: list[str] = [f"mtllib {output_mtl_path.name}", "o input_mesh", "usemtl mesh_white", "s off"]
    for x, y, z in vertices:
        lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
    for nx, ny, nz in face_normals:
        lines.append(f"vn {nx:.9g} {ny:.9g} {nz:.9g}")
    for normal_index, (f0, f1, f2) in enumerate(faces, start=1):
        i0 = int(f0) + 1
        i1 = int(f1) + 1
        i2 = int(f2) + 1
        lines.append(f"f {i0}//{normal_index} {i1}//{normal_index} {i2}//{normal_index}")

    lines.append("o spiral_polyline")
    lines.append("usemtl spiral_line")
    vertex_offset = int(vertices.shape[0])
    for x, y, z in polyline:
        lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
    indices = " ".join(str(vertex_offset + i) for i in range(1, polyline.shape[0] + 1))
    lines.append(f"l {indices}")

    output_obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("Axis vector must be non-zero.")
    return axis / norm


def resolve_axis(mesh: trimesh.Trimesh, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    if args.axis_vector is not None:
        axis = normalize_axis(np.array(args.axis_vector, dtype=float))
        source = "axis-vector"
    elif args.axis_mode == "manual":
        raise ValueError("--axis-mode manual requires --axis-vector.")
    elif args.axis_mode == "inertia":
        principal = np.asarray(mesh.principal_inertia_vectors, dtype=float)
        axis = normalize_axis(principal[0])
        source = "inertia"
    else:
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        axis = axis_map[args.axis]
        source = f"bbox:{args.axis}"

    if args.axis_origin is not None:
        origin = np.array(args.axis_origin, dtype=float)
        source = f"{source}+origin"
    else:
        origin = np.array(mesh.bounding_box.centroid, dtype=float)
    return axis, origin, source
