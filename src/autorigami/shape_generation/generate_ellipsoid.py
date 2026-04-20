"""Generate a 3D ellipsoid mesh using trimesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


def generate_ellipsoid_mesh(
    radius_x: float = 1.5,
    radius_y: float = 1.0,
    radius_z: float = 0.8,
    subdivisions: int = 4,
) -> trimesh.Trimesh:
    """Create an ellipsoid mesh by scaling an icosphere.

    Args:
        radius_x: Semi-axis length along X.
        radius_y: Semi-axis length along Y.
        radius_z: Semi-axis length along Z.
        subdivisions: Icosphere subdivision level (higher = smoother).
    """
    radii = (radius_x, radius_y, radius_z)
    if any(r <= 0 for r in radii):
        raise ValueError("All radii must be > 0.")
    if subdivisions < 0:
        raise ValueError("subdivisions must be >= 0.")

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    mesh.apply_scale(radii)
    return mesh


def export_ellipsoid(
    output_path: str | Path,
    radius_x: float = 1.5,
    radius_y: float = 1.0,
    radius_z: float = 0.8,
    subdivisions: int = 4,
) -> Path:
    """Generate and export an ellipsoid mesh to disk."""
    path = Path(output_path)
    mesh = generate_ellipsoid_mesh(
        radius_x=radius_x,
        radius_y=radius_y,
        radius_z=radius_z,
        subdivisions=subdivisions,
    )
    mesh.export(path)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an ellipsoid mesh.")
    parser.add_argument(
        "--out", default="outputs/ellipsoid.stl", help="Output mesh file path."
    )
    parser.add_argument("--rx", type=float, default=1.5, help="Radius along X axis.")
    parser.add_argument("--ry", type=float, default=1.0, help="Radius along Y axis.")
    parser.add_argument("--rz", type=float, default=0.8, help="Radius along Z axis.")
    parser.add_argument(
        "--subdivisions",
        type=int,
        default=4,
        help="Icosphere subdivisions (higher = smoother mesh).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_path = export_ellipsoid(
        output_path=args.out,
        radius_x=args.rx,
        radius_y=args.ry,
        radius_z=args.rz,
        subdivisions=args.subdivisions,
    )
    print(f"Saved ellipsoid mesh to: {output_path}")


if __name__ == "__main__":
    main()
