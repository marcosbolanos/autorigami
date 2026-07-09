import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import trimesh

from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.types import Polyline

LIGHTWEIGHT_GLB_MAX_BYTES = 1_000_000
DNA_BASE_PAIRS_PER_TURN = 10.5
DNA_BASE_PAIR_TWIST_RADIANS = 2.0 * math.pi / DNA_BASE_PAIRS_PER_TURN

ADENINE_COLOR = np.array([128, 58, 180, 255], dtype=np.uint8)
THYMINE_COLOR = np.array([198, 84, 205, 255], dtype=np.uint8)
CYTOSINE_COLOR = np.array([88, 64, 205, 255], dtype=np.uint8)
GUANINE_COLOR = np.array([174, 118, 230, 255], dtype=np.uint8)
STRAND_COLOR = np.array([216, 216, 224, 255], dtype=np.uint8)

BASE_PAIR_COMPLEMENTS = (
    (ADENINE_COLOR, THYMINE_COLOR),
    (THYMINE_COLOR, ADENINE_COLOR),
    (CYTOSINE_COLOR, GUANINE_COLOR),
    (GUANINE_COLOR, CYTOSINE_COLOR),
)


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


def _polyline_frames(polyline: Polyline) -> tuple[Polyline, Polyline, Polyline]:
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    assert len(polyline) >= 2, "polyline must contain at least 2 points"

    tangents = np.gradient(polyline, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    assert np.all(tangent_norms > 0.0), (
        "polyline must not contain repeated neighboring points"
    )
    tangents = (tangents / tangent_norms).astype(np.float32)

    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(tangents[0], reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    normal = np.cross(tangents[0], reference)
    normal = normal / np.linalg.norm(normal)
    binormal = np.cross(tangents[0], normal)
    normals = [normal.astype(np.float32)]
    binormals = [binormal.astype(np.float32)]

    for tangent in tangents[1:]:
        normal = normal - tangent * np.dot(normal, tangent)
        normal_norm = np.linalg.norm(normal)
        assert normal_norm > 0.0, "polyline frame became degenerate"
        normal = normal / normal_norm
        binormal = np.cross(tangent, normal)
        normals.append(normal.astype(np.float32))
        binormals.append(binormal.astype(np.float32))

    return tangents, np.vstack(normals), np.vstack(binormals)


def dna_molecule_line_segments_from_base_pair_centers(
    base_pair_centers: Polyline,
    dna_molecule_radius: float,
    random_seed: int = 0,
) -> tuple[Polyline, Polyline, npt.NDArray[np.uint8]]:
    assert base_pair_centers.ndim == 2 and base_pair_centers.shape[1] == 3, (
        "base_pair_centers must have shape (n, 3)"
    )
    assert len(base_pair_centers) >= 2, (
        "base_pair_centers must contain at least 2 points"
    )
    assert dna_molecule_radius > 0.0, "dna_molecule_radius must be positive"

    _, normals, binormals = _polyline_frames(base_pair_centers)
    base_pair_angles = (
        np.arange(len(base_pair_centers), dtype=np.float32) * DNA_BASE_PAIR_TWIST_RADIANS
    )
    radial_directions = (
        np.cos(base_pair_angles)[:, None] * normals
        + np.sin(base_pair_angles)[:, None] * binormals
    ).astype(np.float32)
    strand_a = base_pair_centers + dna_molecule_radius * radial_directions
    strand_b = base_pair_centers - dna_molecule_radius * radial_directions

    strand_segment_count = len(base_pair_centers) - 1
    base_pair_count = len(base_pair_centers)
    segment_count = 2 * strand_segment_count + 2 * base_pair_count
    starts = np.empty((segment_count, 3), dtype=np.float32)
    ends = np.empty((segment_count, 3), dtype=np.float32)
    colors = np.empty((segment_count, 4), dtype=np.uint8)

    cursor = 0
    starts[cursor : cursor + strand_segment_count] = strand_a[:-1]
    ends[cursor : cursor + strand_segment_count] = strand_a[1:]
    colors[cursor : cursor + strand_segment_count] = STRAND_COLOR
    cursor += strand_segment_count

    starts[cursor : cursor + strand_segment_count] = strand_b[:-1]
    ends[cursor : cursor + strand_segment_count] = strand_b[1:]
    colors[cursor : cursor + strand_segment_count] = STRAND_COLOR
    cursor += strand_segment_count

    rng = np.random.default_rng(random_seed)
    base_pair_indices = rng.integers(0, len(BASE_PAIR_COMPLEMENTS), size=base_pair_count)
    complement_colors = np.array(BASE_PAIR_COMPLEMENTS, dtype=np.uint8)
    base_pair_colors = complement_colors[base_pair_indices]

    starts[cursor : cursor + base_pair_count] = strand_a
    ends[cursor : cursor + base_pair_count] = base_pair_centers
    colors[cursor : cursor + base_pair_count] = base_pair_colors[:, 0]
    cursor += base_pair_count

    starts[cursor : cursor + base_pair_count] = base_pair_centers
    ends[cursor : cursor + base_pair_count] = strand_b
    colors[cursor : cursor + base_pair_count] = base_pair_colors[:, 1]

    return starts, ends, colors


def _low_poly_segment_mesh(
    starts: Polyline,
    ends: Polyline,
    colors: npt.NDArray[np.uint8],
    radius: float,
    radial_sections: int,
) -> trimesh.Trimesh:
    assert starts.ndim == 2 and starts.shape[1] == 3, "starts must have shape (n, 3)"
    assert ends.shape == starts.shape, "ends must have the same shape as starts"
    assert colors.shape == (len(starts), 4), "colors must have shape (n, 4)"
    assert radius > 0.0, "radius must be positive"
    assert radial_sections >= 3, "radial_sections must be at least 3"

    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    assert np.all(lengths > 0.0), "segment endpoints must differ"
    tangents = directions / lengths

    references = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(starts), 1))
    near_vertical = np.abs(np.sum(tangents * references, axis=1)) > 0.9
    references[near_vertical] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    normals = np.cross(tangents, references)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    binormals = np.cross(tangents, normals)

    angles = np.linspace(
        0.0, 2.0 * math.pi, radial_sections, endpoint=False, dtype=np.float32
    )
    offsets = radius * (
        np.cos(angles)[None, :, None] * normals[:, None, :]
        + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )
    rings = np.stack((starts[:, None, :] + offsets, ends[:, None, :] + offsets), axis=1)
    vertices = rings.reshape((-1, 3)).astype(np.float32)

    segment_offsets = (
        np.arange(len(starts), dtype=np.int64)[:, None] * (2 * radial_sections)
    )
    section_indices = np.arange(radial_sections, dtype=np.int64)[None, :]
    next_section_indices = (section_indices + 1) % radial_sections
    start_ring = segment_offsets + section_indices
    next_start_ring = segment_offsets + next_section_indices
    end_ring = segment_offsets + radial_sections + section_indices
    next_end_ring = segment_offsets + radial_sections + next_section_indices
    faces = np.stack(
        (
            np.stack((start_ring, end_ring, next_end_ring), axis=2),
            np.stack((start_ring, next_end_ring, next_start_ring), axis=2),
        ),
        axis=2,
    ).reshape((-1, 3))

    vertex_colors = np.repeat(colors, 2 * radial_sections, axis=0)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        process=False,
    )


def dna_molecule_mesh_from_base_pair_centers(
    base_pair_centers: Polyline,
    dna_molecule_radius: float,
    segment_radius: float,
    radial_sections: int = 4,
    random_seed: int = 0,
) -> trimesh.Trimesh:
    assert segment_radius > 0.0, "segment_radius must be positive"
    assert radial_sections >= 3, "radial_sections must be at least 3"
    starts, ends, colors = dna_molecule_line_segments_from_base_pair_centers(
        base_pair_centers=base_pair_centers,
        dna_molecule_radius=dna_molecule_radius,
        random_seed=random_seed,
    )
    return _low_poly_segment_mesh(
        starts=starts,
        ends=ends,
        colors=colors,
        radius=segment_radius,
        radial_sections=radial_sections,
    )


def dna_molecule_mesh_from_polyline(
    polyline: Polyline,
    distance_between_base_pairs: float,
    dna_molecule_radius: float,
    segment_radius: float,
    radial_sections: int = 4,
    random_seed: int = 0,
) -> trimesh.Trimesh:
    assert distance_between_base_pairs > 0.0, (
        "distance_between_base_pairs must be positive"
    )
    base_pair_centers = reparametrize_arc_length(
        polyline, distance_between_base_pairs
    ).astype(np.float32)
    return dna_molecule_mesh_from_base_pair_centers(
        base_pair_centers=base_pair_centers,
        dna_molecule_radius=dna_molecule_radius,
        segment_radius=segment_radius,
        radial_sections=radial_sections,
        random_seed=random_seed,
    )


def save_dna_molecule_glb(
    base_pair_centers: Polyline,
    path: Path,
    dna_molecule_radius: float,
    segment_radius: float,
    radial_sections: int = 4,
) -> Path:
    dna_mesh = dna_molecule_mesh_from_base_pair_centers(
        base_pair_centers=base_pair_centers,
        dna_molecule_radius=dna_molecule_radius,
        segment_radius=segment_radius,
        radial_sections=radial_sections,
    )
    dna_mesh.export(path)
    return path


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
