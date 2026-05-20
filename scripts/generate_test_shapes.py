from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def _rotation_z(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _save_ellipsoid(path: Path, scale: tuple[float, float, float], rotation: np.ndarray | None = None) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    if rotation is not None:
        vertices = vertices @ rotation.T
    mesh.vertices = vertices
    mesh.export(path)


def main() -> None:
    out_dir = Path("assets/test_shapes")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_ellipsoid(out_dir / "sphere.obj", (1.0, 1.0, 1.0))
    _save_ellipsoid(out_dir / "ellipsoid_long.obj", (1.8, 1.0, 0.75))
    _save_ellipsoid(out_dir / "ellipsoid_flat.obj", (1.4, 1.2, 0.55))
    _save_ellipsoid(out_dir / "ellipsoid_rotated.obj", (1.8, 1.0, 0.75), _rotation_z(np.deg2rad(32.0)))


if __name__ == "__main__":
    main()
