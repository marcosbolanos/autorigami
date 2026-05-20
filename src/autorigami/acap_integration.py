from __future__ import annotations

from pathlib import Path
import re
import subprocess

import numpy as np

from autorigami._native import compute_acap_toolpath_stats

_GCODE_COORDINATE_RE = re.compile(r"([XYZ])([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")


def acap_submodule_status(repo_root: Path | None = None) -> dict[str, object]:
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    submodule_path = root / "external" / "acap"
    status: dict[str, object] = {
        "name": "acap",
        "repository": "https://github.com/marcosbolanos/acap.git",
        "submodule_path": str(submodule_path),
        "available": submodule_path.exists(),
        "native_adapter_built_by_autorigami": True,
        "upstream_algorithm_built_by_autorigami": False,
        "integration_mode": "submodule_plus_native_toolpath_stats",
        "notes": (
            "ACAP is vendored for comparison/interoperability. Autorigami builds a native pybind "
            "adapter for ACAP-style toolpath stats, but does not build or execute ACAP's upstream "
            "algorithm yet because its entrypoint has external native dependencies and hard-coded "
            "local paths."
        ),
    }
    if not submodule_path.exists():
        status["head_commit"] = None
        return status

    result = subprocess.run(
        ["git", "-C", str(submodule_path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status["head_commit"] = result.stdout.strip() if result.returncode == 0 else None
    return status


def load_acap_polyline(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _load_obj_vertices(path)
    if suffix in {".gcode", ".gc"}:
        return _load_gcode_motion_points(path)
    raise ValueError(f"Unsupported ACAP path format: {path.suffix}")


def compute_toolpath_stats(
    *,
    points: np.ndarray,
    minimum_separation_world: float,
    nonlocal_window_world: float,
    world_to_nm: float,
    nonlocal_violation_count: int | None = None,
    minimum_checked_distance_world: float | None = None,
    distribution_sample_count: int = 1200,
) -> dict[str, object]:
    checked_points = _as_polyline_points(points)
    length_world = _polyline_length(checked_points)
    if nonlocal_violation_count is None or minimum_checked_distance_world is None:
        native_stats = compute_acap_toolpath_stats(
            checked_points,
            minimum_separation_world,
            nonlocal_window_world,
        )
        length_world = float(native_stats["length_world"])
        nonlocal_violation_count = int(native_stats["nonlocal_violation_count"])
        minimum_checked_distance_world = float(native_stats["minimum_checked_distance_world"])
    distribution = _sampled_nonlocal_distance_distribution(
        checked_points,
        nonlocal_window_world=nonlocal_window_world,
        sample_count=distribution_sample_count,
    )
    min_distance_world = float(minimum_checked_distance_world)
    return {
        "format_family": "acap_polyline_toolpath",
        "point_count": int(checked_points.shape[0]),
        "length_world": length_world,
        "length_nm": length_world * world_to_nm,
        "minimum_separation_world": minimum_separation_world,
        "minimum_separation_nm": minimum_separation_world * world_to_nm,
        "nonlocal_window_world": nonlocal_window_world,
        "nonlocal_window_nm": nonlocal_window_world * world_to_nm,
        "nonlocal_violation_count": int(nonlocal_violation_count),
        "minimum_checked_distance_world": min_distance_world,
        "minimum_checked_distance_nm": min_distance_world * world_to_nm,
        "sampled_nonlocal_distance_distribution_nm": {
            key: value * world_to_nm for key, value in distribution.items()
        },
    }


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid OBJ vertex line in {path}: {raw_line}")
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return _as_polyline_points(np.asarray(vertices, dtype=np.float64))


def _load_gcode_motion_points(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    current = np.zeros(3, dtype=np.float64)
    initialized = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", maxsplit=1)[0].strip().upper()
        if not (line.startswith("G0") or line.startswith("G1")):
            continue
        next_point = current.copy()
        found_axis = False
        for axis, value in _GCODE_COORDINATE_RE.findall(line):
            axis_index = {"X": 0, "Y": 1, "Z": 2}[axis]
            next_point[axis_index] = float(value)
            found_axis = True
        if not found_axis:
            continue
        current = next_point
        if initialized and np.allclose(points[-1], current):
            continue
        points.append(current.tolist())
        initialized = True
    return _as_polyline_points(np.asarray(points, dtype=np.float64))


def _as_polyline_points(points: np.ndarray) -> np.ndarray:
    checked = np.asarray(points, dtype=np.float64)
    if checked.ndim != 2 or checked.shape[1] != 3:
        raise ValueError(f"Polyline points must have shape (n, 3), got {checked.shape}")
    if checked.shape[0] < 2:
        raise ValueError("Polyline requires at least two points")
    if not np.all(np.isfinite(checked)):
        raise ValueError("Polyline points must be finite")
    return checked


def _polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))


def _sampled_nonlocal_distance_distribution(
    points: np.ndarray,
    *,
    nonlocal_window_world: float,
    sample_count: int,
) -> dict[str, float]:
    if points.shape[0] <= sample_count:
        sampled = points
    else:
        indices = np.linspace(0, points.shape[0] - 1, sample_count, dtype=np.int64)
        sampled = points[indices]

    arclength = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.cumsum(np.linalg.norm(sampled[1:] - sampled[:-1], axis=1)),
        ]
    )
    distances: list[np.ndarray] = []
    for index in range(sampled.shape[0] - 1):
        arc_delta = arclength[index + 1 :] - arclength[index]
        mask = arc_delta > nonlocal_window_world
        if not np.any(mask):
            continue
        delta = sampled[index + 1 :][mask] - sampled[index]
        distances.append(np.linalg.norm(delta, axis=1))

    if not distances:
        return {
            "minimum": float("inf"),
            "p05": float("inf"),
            "p25": float("inf"),
            "p50": float("inf"),
            "p75": float("inf"),
            "p95": float("inf"),
        }

    flat = np.concatenate(distances)
    percentiles = np.percentile(flat, [5.0, 25.0, 50.0, 75.0, 95.0])
    return {
        "minimum": float(np.min(flat)),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }
