import math
from pathlib import Path

import numpy as np
import trimesh

from autorigami.types import Polyline

LIGHTWEIGHT_GLB_MAX_BYTES = 1_000_000


def _resample_polyline(polyline: Polyline, n_samples: int) -> Polyline:
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    assert n_samples >= 2, "n_samples must be at least 2"

    segment_lengths = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_lengths = np.linspace(
        0.0, cumulative_lengths[-1], n_samples, dtype=np.float32
    )
    points = np.column_stack(
        [
            np.interp(sample_lengths, cumulative_lengths, polyline[:, axis])
            for axis in range(3)
        ]
    )
    return points.astype(np.float32)


def _tube_mesh_from_polyline(
    polyline: Polyline,
    radius: float,
    radial_sections: int,
) -> trimesh.Trimesh:
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    assert len(polyline) >= 2, "polyline must contain at least 2 points"
    assert radius > 0, "radius must be positive"
    assert radial_sections >= 3, "radial_sections must be at least 3"

    tangents = np.gradient(polyline, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    assert np.all(tangent_norms > 0), (
        "polyline must not contain repeated neighboring points"
    )
    tangents = tangents / tangent_norms

    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(tangents[0], reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    normal = np.cross(tangents[0], reference)
    normal = normal / np.linalg.norm(normal)
    binormal = np.cross(tangents[0], normal)

    angles = np.linspace(
        0.0, 2.0 * math.pi, radial_sections, endpoint=False, dtype=np.float32
    )
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)

    rings: list[np.ndarray] = []
    for index, tangent in enumerate(tangents):
        if index > 0:
            normal = normal - tangent * np.dot(normal, tangent)
            normal_norm = np.linalg.norm(normal)
            assert normal_norm > 0, "polyline frame became degenerate"
            normal = normal / normal_norm
            binormal = np.cross(tangent, normal)
        ring = (
            polyline[index]
            + radius * unit_circle[:, 0, None] * normal
            + radius * unit_circle[:, 1, None] * binormal
        )
        rings.append(ring)

    vertices = np.vstack(rings).astype(np.float32)
    faces: list[list[int]] = []
    for ring_index in range(len(polyline) - 1):
        ring_start = ring_index * radial_sections
        next_ring_start = (ring_index + 1) * radial_sections
        for section_index in range(radial_sections):
            next_section_index = (section_index + 1) % radial_sections
            faces.append(
                [
                    ring_start + section_index,
                    next_ring_start + section_index,
                    next_ring_start + next_section_index,
                ]
            )
            faces.append(
                [
                    ring_start + section_index,
                    next_ring_start + next_section_index,
                    ring_start + next_section_index,
                ]
            )

    start_center = len(vertices)
    end_center = start_center + 1
    vertices = np.vstack((vertices, polyline[0], polyline[-1])).astype(np.float32)
    end_ring_start = (len(polyline) - 1) * radial_sections
    for section_index in range(radial_sections):
        next_section_index = (section_index + 1) % radial_sections
        faces.append([start_center, next_section_index, section_index])
        faces.append(
            [
                end_center,
                end_ring_start + section_index,
                end_ring_start + next_section_index,
            ]
        )

    vertex_colors = np.tile(
        np.array([[190, 190, 190, 255]], dtype=np.uint8), (vertices.shape[0], 1)
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(faces, dtype=np.int64),
        vertex_colors=vertex_colors,
        process=False,
    )


def save_lightweight_glb(
    polyline: Polyline,
    path: Path,
    radius: float,
    max_bytes: int = LIGHTWEIGHT_GLB_MAX_BYTES,
) -> Path:
    assert max_bytes > 0, "max_bytes must be positive"

    quality_levels = ((1400, 8), (1000, 8), (800, 7), (600, 6), (450, 6))
    for n_samples, radial_sections in quality_levels:
        lightweight_polyline = _resample_polyline(polyline, n_samples)
        lightweight_mesh = _tube_mesh_from_polyline(
            lightweight_polyline, radius, radial_sections
        )
        lightweight_mesh.export(path)
        if path.stat().st_size < max_bytes:
            return path

    file_size = path.stat().st_size
    raise AssertionError(
        f"lightweight GLB is {file_size} bytes, expected under {max_bytes} bytes"
    )
