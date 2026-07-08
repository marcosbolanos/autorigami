import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pyvista as pv
import numpy as np
import trimesh

from autorigami.types import Vector3, Polyline

LIGHTWEIGHT_GLB_MAX_BYTES = 1_000_000


# Generic spiral class, get_point() is the parametrization
@dataclass
class SpiralObject(ABC):
    starting_angle: float
    winding_frequency: float
    length: float

    # The parametrization function, to be defined by inheriting classes
    @abstractmethod
    def get_point(self, position: float) -> Vector3:
        ...

    # Discretization is the same for every type of spiral
    def discretize(self, n_samples: int) -> tuple[Polyline, float]:
        positions = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
        polyline: Polyline = np.array([self.get_point(position) for position in positions])
        # We track the final angle to chain together multiple spirals
        final_angle = float(self.starting_angle + 2.0 * math.pi * self.winding_frequency * positions[-1])
        return polyline, final_angle


# Base of the spiral, this is a simple cylinder-shaped one
@dataclass
class SpiralBase(SpiralObject):
    starting_angle: float
    radius: float
    winding_frequency: float
    length: float

    def get_point(self, position: float) -> Vector3:
        assert 0 <= position <= 1, "position must be between 0 and 1"

        # Angle is a linear function of position
        angle = self.starting_angle + 2.0 * math.pi * self.winding_frequency * position
        x = self.radius * math.cos(angle)
        y = self.radius * math.sin(angle)
        z = self.length * position
        point = np.array([x, y, z], dtype=np.float32)
        return point

@dataclass
class MiddleSegment(SpiralObject):
    starting_angle: float
    starting_coords: Vector3
    starting_x_radius: float
    x_radius_increase_factor: float
    y_radius: float
    winding_frequency: float
    length: float

    def get_point(self, position: float) -> Vector3:
        assert 0 <= position <= 1, "position must be between 0 and 1"

        local_angle = 2.0 * math.pi * self.winding_frequency * position
        x_radius = self.starting_x_radius * (1.0 + position * (self.x_radius_increase_factor - 1.0))
        inward_position = min(1.0, max(0.0, (position - 0.2) / 0.8))
        q = (local_angle + 0.25 * math.pi) % (2.0 * math.pi)
        side_center_x = x_radius - self.y_radius

        if q < 0.5 * math.pi:
            arc_angle = -0.5 * math.pi + 2.0 * q
            x = side_center_x + self.y_radius * math.cos(arc_angle)
            y = self.y_radius * math.sin(arc_angle)
        elif q < math.pi:
            u = (q - 0.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            x = side_center_x * (1.0 - 2.0 * u)
            y = self.y_radius * (1.0 - 0.95 * inward_position * math.sin(math.pi * u) ** 2)
        elif q < 1.5 * math.pi:
            arc_angle = 0.5 * math.pi + 2.0 * (q - math.pi)
            x = -side_center_x + self.y_radius * math.cos(arc_angle)
            y = self.y_radius * math.sin(arc_angle)
        else:
            u = (q - 1.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            x = side_center_x * (-1.0 + 2.0 * u)
            y = -self.y_radius * (1.0 - 0.95 * inward_position * math.sin(math.pi * u) ** 2)

        x, y = (
            x * math.cos(self.starting_angle) - y * math.sin(self.starting_angle),
            x * math.sin(self.starting_angle) + y * math.cos(self.starting_angle),
        )
        start_x = self.starting_x_radius
        start_y = 0.0
        start_x, start_y = (
            start_x * math.cos(self.starting_angle) - start_y * math.sin(self.starting_angle),
            start_x * math.sin(self.starting_angle) + start_y * math.cos(self.starting_angle),
        )
        center = self.starting_coords - np.array([start_x, start_y, 0.0], dtype=np.float32)
        z = self.length * position
        point = center + np.array([x, y, z], dtype=np.float32)
        return point


@dataclass
class TopSegment(SpiralObject):
    starting_angle: float
    starting_coords: Vector3
    starting_x_radius: float
    x_radius_increase_factor: float
    y_radius: float
    winding_frequency: float
    length: float
    phase_offset: float

    def get_point(self, position: float) -> Vector3:
        assert 0 <= position <= 1, "position must be between 0 and 1"

        local_angle = self.phase_offset + 2.0 * math.pi * self.winding_frequency * position
        x_radius = self.starting_x_radius * (1.0 + position * (self.x_radius_increase_factor - 1.0))
        q = (local_angle + 0.25 * math.pi) % (2.0 * math.pi)
        side_center_x = x_radius - self.y_radius
        geometry_position = position * position * (3.0 - 2.0 * position)
        join_angle = 0.5 * math.pi + 0.5 * math.pi * geometry_position
        bridge_y_scale = 0.65

        if q < 0.5 * math.pi:
            old_arc_angle = -0.5 * math.pi + 2.0 * q
            new_arc_angle = -join_angle + (q / (0.5 * math.pi)) * 2.0 * join_angle
            old_x = side_center_x + self.y_radius * math.cos(old_arc_angle)
            old_y = self.y_radius * math.sin(old_arc_angle)
            new_x = side_center_x + self.y_radius * math.cos(new_arc_angle)
            new_y = self.y_radius * math.sin(new_arc_angle)
            x = old_x * (1.0 - geometry_position) + new_x * geometry_position
            y = old_y * (1.0 - geometry_position) + new_y * geometry_position
        elif q < math.pi:
            u = (q - 0.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            old_x = side_center_x * (1.0 - 2.0 * u)
            old_y = self.y_radius * (1.0 - 0.95 * math.sin(math.pi * u) ** 2)
            if math.isclose(math.cos(join_angle), 0.0, abs_tol=1e-6):
                new_x = old_x
                new_y = old_y
            else:
                bridge_start_x = side_center_x + self.y_radius * math.cos(join_angle)
                bridge_start_y = self.y_radius * math.sin(join_angle)
                bridge_start_angle = math.atan(bridge_y_scale * math.tan(join_angle))
                bridge_x_radius = bridge_start_x / math.cos(bridge_start_angle)
                bridge_y_radius = bridge_y_scale * bridge_x_radius
                bridge_center_y = bridge_start_y - bridge_y_radius * math.sin(bridge_start_angle)
                bridge_angle = bridge_start_angle + u * (-math.pi - 2.0 * bridge_start_angle)
                new_x = bridge_x_radius * math.cos(bridge_angle)
                new_y = bridge_center_y + bridge_y_radius * math.sin(bridge_angle)
            x = old_x * (1.0 - geometry_position) + new_x * geometry_position
            y = old_y * (1.0 - geometry_position) + new_y * geometry_position
        elif q < 1.5 * math.pi:
            old_arc_angle = 0.5 * math.pi + 2.0 * (q - math.pi)
            new_arc_angle = math.pi - join_angle + ((q - math.pi) / (0.5 * math.pi)) * 2.0 * join_angle
            old_x = -side_center_x + self.y_radius * math.cos(old_arc_angle)
            old_y = self.y_radius * math.sin(old_arc_angle)
            new_x = -side_center_x + self.y_radius * math.cos(new_arc_angle)
            new_y = self.y_radius * math.sin(new_arc_angle)
            x = old_x * (1.0 - geometry_position) + new_x * geometry_position
            y = old_y * (1.0 - geometry_position) + new_y * geometry_position
        else:
            u = (q - 1.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            old_x = side_center_x * (-1.0 + 2.0 * u)
            old_y = -self.y_radius * (1.0 - 0.95 * math.sin(math.pi * u) ** 2)
            if math.isclose(math.cos(join_angle), 0.0, abs_tol=1e-6):
                new_x = old_x
                new_y = old_y
            else:
                bridge_start_x = side_center_x + self.y_radius * math.cos(join_angle)
                bridge_start_y = self.y_radius * math.sin(join_angle)
                bridge_start_angle = math.atan(bridge_y_scale * math.tan(join_angle))
                bridge_x_radius = bridge_start_x / math.cos(bridge_start_angle)
                bridge_y_radius = bridge_y_scale * bridge_x_radius
                bridge_center_y = -bridge_start_y + bridge_y_radius * math.sin(bridge_start_angle)
                bridge_angle = math.pi + bridge_start_angle + u * (-math.pi - 2.0 * bridge_start_angle)
                new_x = bridge_x_radius * math.cos(bridge_angle)
                new_y = bridge_center_y + bridge_y_radius * math.sin(bridge_angle)
            x = old_x * (1.0 - geometry_position) + new_x * geometry_position
            y = old_y * (1.0 - geometry_position) + new_y * geometry_position

        x, y = (
            x * math.cos(self.starting_angle) - y * math.sin(self.starting_angle),
            x * math.sin(self.starting_angle) + y * math.cos(self.starting_angle),
        )

        start_q = (self.phase_offset + 0.25 * math.pi) % (2.0 * math.pi)
        start_side_center_x = self.starting_x_radius - self.y_radius
        start_join_angle = 0.5 * math.pi
        if start_q < 0.5 * math.pi:
            start_arc_angle = -start_join_angle + (start_q / (0.5 * math.pi)) * 2.0 * start_join_angle
            start_x = start_side_center_x + self.y_radius * math.cos(start_arc_angle)
            start_y = self.y_radius * math.sin(start_arc_angle)
        elif start_q < math.pi:
            start_u = (start_q - 0.5 * math.pi) / (0.5 * math.pi)
            start_u = start_u * start_u * (3.0 - 2.0 * start_u)
            start_x = start_side_center_x * (1.0 - 2.0 * start_u)
            start_y = self.y_radius * (1.0 - 0.95 * math.sin(math.pi * start_u) ** 2)
        elif start_q < 1.5 * math.pi:
            start_arc_angle = math.pi - start_join_angle + ((start_q - math.pi) / (0.5 * math.pi)) * 2.0 * start_join_angle
            start_x = -start_side_center_x + self.y_radius * math.cos(start_arc_angle)
            start_y = self.y_radius * math.sin(start_arc_angle)
        else:
            start_u = (start_q - 1.5 * math.pi) / (0.5 * math.pi)
            start_u = start_u * start_u * (3.0 - 2.0 * start_u)
            start_x = start_side_center_x * (-1.0 + 2.0 * start_u)
            start_y = -self.y_radius * (1.0 - 0.95 * math.sin(math.pi * start_u) ** 2)

        start_x, start_y = (
            start_x * math.cos(self.starting_angle) - start_y * math.sin(self.starting_angle),
            start_x * math.sin(self.starting_angle) + start_y * math.cos(self.starting_angle),
        )
        center = self.starting_coords - np.array([start_x, start_y, 0.0], dtype=np.float32)
        z = self.length * position
        point = center + np.array([x, y, z], dtype=np.float32)
        return point


def generate_full_spiral():
    nm_per_full_turn = 2.6
    length = 40
    radius = 16
    winding_frequency = length / nm_per_full_turn
    
    # unit: nm
    base = SpiralBase(
        starting_angle=0,
        radius=radius,
        winding_frequency=winding_frequency,
        length=length
    )
    polyline, current_angle = base.discretize(10000)
    current_coords = polyline[-1]

    middle_scale = 1
    middle_segment = MiddleSegment(
        starting_angle=current_angle,
        starting_coords=current_coords,
        winding_frequency=winding_frequency * middle_scale,
        length=length * middle_scale,
        starting_x_radius=radius,
        x_radius_increase_factor=3,
        y_radius=radius
    )
    new_polyline, current_angle = middle_segment.discretize(10000)
    polyline = np.concatenate((polyline, new_polyline[1:]), axis=0)
    current_coords = polyline[-1]

    top_segment = TopSegment(
        starting_angle=middle_segment.starting_angle,
        starting_coords=current_coords,
        winding_frequency=0.5 * winding_frequency * middle_scale,
        length=length * middle_scale,
        starting_x_radius=radius * middle_segment.x_radius_increase_factor,
        x_radius_increase_factor=math.sqrt(3),
        y_radius=radius,
        phase_offset=current_angle - middle_segment.starting_angle,
    )
    new_polyline, current_angle = top_segment.discretize(10000)
    polyline = np.concatenate((polyline, new_polyline[1:]), axis=0)
    
    return polyline


def _resample_polyline(polyline: Polyline, n_samples: int) -> Polyline:
    assert polyline.ndim == 2 and polyline.shape[1] == 3, "polyline must have shape (n, 3)"
    assert n_samples >= 2, "n_samples must be at least 2"

    segment_lengths = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_lengths = np.linspace(0.0, cumulative_lengths[-1], n_samples, dtype=np.float32)
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
    assert polyline.ndim == 2 and polyline.shape[1] == 3, "polyline must have shape (n, 3)"
    assert len(polyline) >= 2, "polyline must contain at least 2 points"
    assert radius > 0, "radius must be positive"
    assert radial_sections >= 3, "radial_sections must be at least 3"

    tangents = np.gradient(polyline, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    assert np.all(tangent_norms > 0), "polyline must not contain repeated neighboring points"
    tangents = tangents / tangent_norms

    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(tangents[0], reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    normal = np.cross(tangents[0], reference)
    normal = normal / np.linalg.norm(normal)
    binormal = np.cross(tangents[0], normal)

    angles = np.linspace(0.0, 2.0 * math.pi, radial_sections, endpoint=False, dtype=np.float32)
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
            faces.append([ring_start + section_index, next_ring_start + section_index, next_ring_start + next_section_index])
            faces.append([ring_start + section_index, next_ring_start + next_section_index, ring_start + next_section_index])

    start_center = len(vertices)
    end_center = start_center + 1
    vertices = np.vstack((vertices, polyline[0], polyline[-1])).astype(np.float32)
    end_ring_start = (len(polyline) - 1) * radial_sections
    for section_index in range(radial_sections):
        next_section_index = (section_index + 1) % radial_sections
        faces.append([start_center, next_section_index, section_index])
        faces.append([end_center, end_ring_start + section_index, end_ring_start + next_section_index])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64), process=False)
    mesh.visual.vertex_colors = np.tile(np.array([[190, 190, 190, 255]], dtype=np.uint8), (mesh.vertices.shape[0], 1))
    return mesh


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
        lightweight_mesh = _tube_mesh_from_polyline(lightweight_polyline, radius, radial_sections)
        lightweight_mesh.export(path)
        if path.stat().st_size < max_bytes:
            return path

    file_size = path.stat().st_size
    raise AssertionError(f"lightweight GLB is {file_size} bytes, expected under {max_bytes} bytes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    dna_molecule_radius = 1.05
    polyline = generate_full_spiral()
    tube = pv.lines_from_points(polyline).tube(radius=dna_molecule_radius)

    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"full_double_spiral_{timestamp}"
        output_dir.mkdir(parents=True)
        np.save(output_dir / "polyline.npy", polyline)
        tube.save(output_dir / "full_double_spiral.stl")
        lightweight_glb_path = save_lightweight_glb(
            polyline,
            output_dir / "full_double_spiral_lightweight.glb",
            dna_molecule_radius,
        )
        print(f"Saved lightweight GLB to {lightweight_glb_path} ({lightweight_glb_path.stat().st_size} bytes)")
        print(output_dir)
        return 0

    plotter = pv.Plotter()
    plotter.add_mesh(tube) # type: ignore
    try:
        plotter.show(interactive_update=True, auto_close=False)
        while not plotter._closed:
            plotter.update(stime=10)
    except KeyboardInterrupt:
        plotter.close()
        return 130
    plotter.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
