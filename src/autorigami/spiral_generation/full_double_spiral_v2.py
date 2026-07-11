import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pyvista as pv
import numpy as np
import numpy.typing as npt

from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.mesh_io import (
    dna_molecule_line_segments_from_base_pair_centers,
    dna_molecule_mesh_from_base_pair_centers,
    save_lightweight_glb,
)
from autorigami.types import Vector3, Polyline

Positions = npt.NDArray[np.float32]


# Generic spiral class, get_points() is the parametrization
@dataclass
class SpiralObject(ABC):
    """
    attributes:
    orientation_angle: angle to start drawing the spiral, in radians
    turns: how many times the spiral turns across its height
    height: how high the spiral goes, in nm

    methods:
    get_points: the parametrization of the spiral, gamma(t)
    """

    orientation_angle: float
    turns: float
    height: float

    @abstractmethod
    def get_points(
        self,
        positions: float | np.float32 | Positions,
    ) -> Vector3 | Polyline:
        """Return points for normalized positions in [0, 1].

        positions: can be either a float or an array of floats. Returns spiral coordinates for every position
        """


# Base of the spiral, this is a simple cylinder-shaped one
@dataclass
class SpiralBase(SpiralObject):
    orientation_angle: float
    radius: float
    turns: float
    height: float

    def get_points(
        self,
        positions: float | np.float32 | Positions,
    ) -> Vector3 | Polyline:
        scalar_input = np.isscalar(positions)
        positions = np.array(
            [positions] if scalar_input else positions, dtype=np.float32
        )
        assert positions.ndim == 1, "positions must be one-dimensional"
        assert np.all((0.0 <= positions) & (positions <= 1.0)), (
            "positions must be between 0 and 1"
        )

        # Angle is a linear function of position
        angle = self.orientation_angle + 2.0 * math.pi * self.turns * positions
        # x and y are radius times sines and cosines of the angle, while z is linear wrt position
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        z = self.height * positions
        polyline: Polyline = np.column_stack((x, y, z)).astype(np.float32)
        return polyline[0] if scalar_input else polyline


@dataclass
class MiddleSegment(SpiralObject):
    orientation_angle: float
    anchor_point: Vector3
    start_x_radius: float
    end_x_radius_scale: float
    y_radius: float
    turns: float
    height: float
    phase_offset: float = 0.0
    inward_position_offset: float = 0.0

    def get_points(
        self,
        positions: float | np.float32 | Positions,
    ) -> Vector3 | Polyline:
        scalar_input = np.isscalar(positions)
        positions = np.array(
            [positions] if scalar_input else positions, dtype=np.float32
        )
        assert positions.ndim == 1, "positions must be one-dimensional"
        assert np.all((0.0 <= positions) & (positions <= 1.0)), (
            "positions must be between 0 and 1"
        )

        local_angle = self.phase_offset + 2.0 * math.pi * self.turns * positions
        x_radius = self.start_x_radius * (
            1.0 + positions * (self.end_x_radius_scale - 1.0)
        )
        inward_position = np.clip(
            (positions + self.inward_position_offset - 0.2) / 0.8, 0.0, 1.0
        )
        q = np.mod(local_angle + 0.25 * math.pi, 2.0 * math.pi)
        side_center_x = x_radius - self.y_radius
        x = np.empty_like(positions)
        y = np.empty_like(positions)

        mask = q < 0.5 * math.pi
        arc_angle = -0.5 * math.pi + 2.0 * q[mask]
        x[mask] = side_center_x[mask] + self.y_radius * np.cos(arc_angle)
        y[mask] = self.y_radius * np.sin(arc_angle)

        mask = (0.5 * math.pi <= q) & (q < math.pi)
        if np.any(mask):
            u = (q[mask] - 0.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            x[mask] = side_center_x[mask] * (1.0 - 2.0 * u)
            y[mask] = self.y_radius * (
                1.0 - 0.95 * inward_position[mask] * np.sin(math.pi * u) ** 2
            )

        mask = (math.pi <= q) & (q < 1.5 * math.pi)
        arc_angle = 0.5 * math.pi + 2.0 * (q[mask] - math.pi)
        x[mask] = -side_center_x[mask] + self.y_radius * np.cos(arc_angle)
        y[mask] = self.y_radius * np.sin(arc_angle)

        mask = 1.5 * math.pi <= q
        if np.any(mask):
            u = (q[mask] - 1.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            x[mask] = side_center_x[mask] * (-1.0 + 2.0 * u)
            y[mask] = -self.y_radius * (
                1.0 - 0.95 * inward_position[mask] * np.sin(math.pi * u) ** 2
            )

        cos_orientation = math.cos(self.orientation_angle)
        sin_orientation = math.sin(self.orientation_angle)
        x, y = (
            x * cos_orientation - y * sin_orientation,
            x * sin_orientation + y * cos_orientation,
        )
        start_q = (self.phase_offset + 0.25 * math.pi) % (2.0 * math.pi)
        start_side_center_x = self.start_x_radius - self.y_radius
        if start_q < 0.5 * math.pi:
            start_arc_angle = -0.5 * math.pi + 2.0 * start_q
            start_x = start_side_center_x + self.y_radius * math.cos(start_arc_angle)
            start_y = self.y_radius * math.sin(start_arc_angle)
        elif start_q < math.pi:
            start_u = (start_q - 0.5 * math.pi) / (0.5 * math.pi)
            start_u = start_u * start_u * (3.0 - 2.0 * start_u)
            start_x = start_side_center_x * (1.0 - 2.0 * start_u)
            start_y = self.y_radius * (1.0 - 0.95 * math.sin(math.pi * start_u) ** 2)
        elif start_q < 1.5 * math.pi:
            start_arc_angle = 0.5 * math.pi + 2.0 * (start_q - math.pi)
            start_x = -start_side_center_x + self.y_radius * math.cos(start_arc_angle)
            start_y = self.y_radius * math.sin(start_arc_angle)
        else:
            start_u = (start_q - 1.5 * math.pi) / (0.5 * math.pi)
            start_u = start_u * start_u * (3.0 - 2.0 * start_u)
            start_x = start_side_center_x * (-1.0 + 2.0 * start_u)
            start_y = -self.y_radius * (1.0 - 0.95 * math.sin(math.pi * start_u) ** 2)
        start_x, start_y = (
            start_x * cos_orientation - start_y * sin_orientation,
            start_x * sin_orientation + start_y * cos_orientation,
        )
        center = self.anchor_point - np.array([start_x, start_y, 0.0], dtype=np.float32)
        z = self.height * positions
        polyline: Polyline = center + np.column_stack((x, y, z)).astype(np.float32)
        return polyline[0] if scalar_input else polyline


@dataclass
class TopSegment(SpiralObject):
    orientation_angle: float
    anchor_point: Vector3
    start_x_radius: float
    end_x_radius_scale: float
    y_radius: float
    turns: float
    height: float
    phase_offset: float
    diagonal_x_offset_scale: float = 0.55
    loop_height_scale: float = 0.16

    def get_points(
        self,
        positions: float | np.float32 | Positions,
    ) -> Vector3 | Polyline:
        scalar_input = np.isscalar(positions)
        positions = np.array(
            [positions] if scalar_input else positions, dtype=np.float32
        )
        assert positions.ndim == 1, "positions must be one-dimensional"
        assert np.all((0.0 <= positions) & (positions <= 1.0)), (
            "positions must be between 0 and 1"
        )

        first_spiral_end = np.float32(0.44)
        loop_end = np.float32(0.56)
        polyline = np.empty((len(positions), 3), dtype=np.float32)

        upward_mask = positions <= first_spiral_end
        if np.any(upward_mask):
            upward_positions = positions[upward_mask] / first_spiral_end
            polyline[upward_mask] = self._upward_spiral_points(upward_positions)

        loop_mask = (first_spiral_end < positions) & (positions < loop_end)
        if np.any(loop_mask):
            loop_positions = (positions[loop_mask] - first_spiral_end) / (
                loop_end - first_spiral_end
            )
            polyline[loop_mask] = self._loop_points(loop_positions)

        downward_mask = loop_end <= positions
        if np.any(downward_mask):
            downward_positions = (positions[downward_mask] - loop_end) / (
                1.0 - loop_end
            )
            polyline[downward_mask] = self._downward_spiral_points(
                downward_positions
            )

        return polyline[0] if scalar_input else polyline

    def _upward_spiral_points(self, positions: Positions) -> Polyline:
        local_points = self._diagonal_spiral_local_points(positions)
        return self._to_world_points(local_points)

    def _loop_points(self, positions: Positions) -> Polyline:
        first_endpoint = self._diagonal_spiral_local_points(
            np.array([1.0], dtype=np.float32)
        )[0]
        mirrored_endpoint = self._mirror_x(first_endpoint)
        midpoint = 0.5 * (first_endpoint + mirrored_endpoint)
        loop_radius_z = self.loop_height_scale * self.height
        angle = math.pi * positions

        x = midpoint[0] + (first_endpoint[0] - midpoint[0]) * np.cos(angle)
        y = first_endpoint[1] * (1.0 - positions) + mirrored_endpoint[1] * positions
        z = midpoint[2] + loop_radius_z * np.sin(angle)

        local_points = np.column_stack((x, y, z)).astype(np.float32)
        return self._to_world_points(local_points)

    def _downward_spiral_points(self, positions: Positions) -> Polyline:
        upward_positions = 1.0 - positions
        local_points = self._diagonal_spiral_local_points(upward_positions)
        local_points = self._mirror_x(local_points)
        return self._to_world_points(local_points)

    def _diagonal_spiral_local_points(self, positions: Positions) -> Polyline:
        angle = self.phase_offset + 2.0 * math.pi * self.turns * positions
        start_angle = self.phase_offset
        diagonal_x_offset = (
            self.diagonal_x_offset_scale * self.start_x_radius * positions
        )

        x = (
            self.y_radius * np.cos(angle)
            - self.y_radius * math.cos(start_angle)
            - diagonal_x_offset
        )
        y = self.y_radius * np.sin(angle) - self.y_radius * math.sin(start_angle)
        z = self.height * positions
        return np.column_stack((x, y, z)).astype(np.float32)

    def _to_world_points(self, local_points: Polyline) -> Polyline:
        cos_orientation = math.cos(self.orientation_angle)
        sin_orientation = math.sin(self.orientation_angle)
        x = (
            local_points[:, 0] * cos_orientation
            - local_points[:, 1] * sin_orientation
        )
        y = (
            local_points[:, 0] * sin_orientation
            + local_points[:, 1] * cos_orientation
        )
        world_points: Polyline = (
            self.anchor_point + np.column_stack((x, y, local_points[:, 2]))
        ).astype(np.float32)
        return world_points

    def _mirror_x(self, points: Vector3 | Polyline) -> Vector3 | Polyline:
        mirrored = np.array(points, dtype=np.float32, copy=True)
        mirrored[..., 0] = self._other_side_x_offset() - mirrored[..., 0]
        return mirrored

    def _other_side_x_offset(self) -> float:
        side_center_x = self.start_x_radius - self.y_radius
        endpoint_x = -side_center_x + self.y_radius * math.cos(self.phase_offset)
        return -2.0 * endpoint_x


def _middle_segment_end_arc_angle(segment: MiddleSegment) -> float:
    q = (
        segment.phase_offset
        + 2.0 * math.pi * segment.turns
        + 0.25 * math.pi
    ) % (2.0 * math.pi)
    if q < 0.5 * math.pi:
        return -0.5 * math.pi + 2.0 * q
    assert math.pi <= q < 1.5 * math.pi, (
        "middle segment must end on a circular side arc"
    )
    return 0.5 * math.pi + 2.0 * (q - math.pi)


def _default_spiral_parameters() -> tuple[float, float, float, Positions]:
    nm_per_full_turn = 2.6
    length = 40.0
    radius = 16.0
    turns = length / nm_per_full_turn
    positions = np.linspace(0.0, 1.0, 10000, dtype=np.float32)
    return length, radius, turns, positions


def _build_default_middle_segment() -> tuple[MiddleSegment, Polyline]:
    length, radius, turns, positions = _default_spiral_parameters()
    base = SpiralBase(
        orientation_angle=0.0,
        radius=radius,
        turns=turns,
        height=length,
    )
    base_polyline = base.get_points(positions)
    assert base_polyline.ndim == 2 and base_polyline.shape[1] == 3, (
        "base_polyline must have shape (n, 3)"
    )
    base_end_angle = float(base.orientation_angle + 2.0 * math.pi * base.turns)
    middle_segment = MiddleSegment(
        orientation_angle=base_end_angle,
        anchor_point=base_polyline[-1],
        turns=turns,
        height=length,
        start_x_radius=radius,
        end_x_radius_scale=3.0,
        y_radius=radius,
    )
    middle_polyline = middle_segment.get_points(positions)
    assert middle_polyline.ndim == 2 and middle_polyline.shape[1] == 3, (
        "middle_polyline must have shape (n, 3)"
    )
    return middle_segment, middle_polyline


def _middle_top_local_to_world(
    local_points: Polyline,
    middle_segment: MiddleSegment,
    middle_polyline: Polyline,
) -> Polyline:
    end_local_point = _middle_segment_end_local_point(middle_segment)
    local_points = (local_points - end_local_point).astype(np.float32)
    cos_orientation = math.cos(middle_segment.orientation_angle)
    sin_orientation = math.sin(middle_segment.orientation_angle)
    x = (
        local_points[:, 0] * cos_orientation
        - local_points[:, 1] * sin_orientation
    )
    y = (
        local_points[:, 0] * sin_orientation
        + local_points[:, 1] * cos_orientation
    )
    world_points: Polyline = (
        middle_polyline[-1] + np.column_stack((x, y, local_points[:, 2]))
    ).astype(np.float32)
    return world_points


def _middle_segment_end_local_point(segment: MiddleSegment) -> Vector3:
    side_center_x = (
        segment.start_x_radius * segment.end_x_radius_scale
        - segment.y_radius
    )
    arc_angle = _middle_segment_end_arc_angle(segment)
    return np.array(
        [
            -side_center_x + segment.y_radius * math.cos(arc_angle),
            segment.y_radius * math.sin(arc_angle),
            0.0,
        ],
        dtype=np.float32,
    )


def generate_middle_top_lid_strands(
    dna_molecule_radius: float = 1.05,
    strand_count: int = 17,
    samples_per_strand: int = 96,
    cover_radius_scale: float = 2.0,
    include_center_strand: bool = True,
) -> list[Polyline]:
    """Generate short blocker strands for the open top of the middle segment.

    The lid is defined in the local top cross-section of the middle segment. Each
    strand spans from the left facing side arc to the right facing side arc, then
    expands past those centerlines by an angle-aware cover distance so the middle
    rods are overlapped rather than merely touched.
    """

    assert dna_molecule_radius > 0.0, "dna_molecule_radius must be positive"
    assert strand_count > 0, "strand_count must be positive"
    assert samples_per_strand >= 2, "samples_per_strand must be at least 2"
    assert cover_radius_scale >= 0.0, "cover_radius_scale must be non-negative"

    middle_segment, middle_polyline = _build_default_middle_segment()
    side_center_x = (
        middle_segment.start_x_radius * middle_segment.end_x_radius_scale
        - middle_segment.y_radius
    )
    arc_radius = middle_segment.y_radius
    cover_radius = cover_radius_scale * dna_molecule_radius
    rod_clearance = 2.0 * dna_molecule_radius
    max_offset = side_center_x - arc_radius
    inner_stop_x = cover_radius
    max_lid_offset = max(rod_clearance, max_offset - dna_molecule_radius)
    side_strand_count = max(1, strand_count // 2)
    offsets = np.linspace(
        rod_clearance,
        max_lid_offset,
        side_strand_count,
        dtype=np.float32,
    )
    lid_strands: list[Polyline] = []

    for offset_float in offsets:
        offset = float(offset_float)
        strand_radius = arc_radius + offset
        exposed_y_radius = math.sqrt(
            max(arc_radius * arc_radius - offset * offset, 0.0)
        )
        if side_center_x - (arc_radius + offset) < inner_stop_x:
            inner_y_radius = math.sqrt(
                max(
                    (arc_radius + offset) ** 2
                    - (side_center_x - inner_stop_x) ** 2,
                    0.0,
                )
            )
            exposed_y_radius = min(exposed_y_radius, inner_y_radius)
        if exposed_y_radius <= 0.0:
            continue

        y = np.linspace(
            -exposed_y_radius,
            exposed_y_radius,
            samples_per_strand,
            dtype=np.float32,
        )
        side_dx = np.sqrt(np.maximum(strand_radius * strand_radius - y**2, 0.0))
        z = np.zeros_like(y)
        left_arc = np.column_stack((-side_center_x + side_dx, y, z)).astype(
            np.float32
        )
        right_arc = np.column_stack((side_center_x - side_dx, y, z)).astype(
            np.float32
        )
        lid_strands.append(
            _middle_top_local_to_world(left_arc, middle_segment, middle_polyline)
        )
        lid_strands.append(
            _middle_top_local_to_world(right_arc, middle_segment, middle_polyline)
        )

    if include_center_strand and strand_count % 2 == 1:
        last_offset = float(offsets[-1])
        remaining_y_radius = math.sqrt(
            max(arc_radius * arc_radius - last_offset * last_offset, 0.0)
        )
        if remaining_y_radius > dna_molecule_radius:
            y = np.linspace(
                -remaining_y_radius,
                remaining_y_radius,
                samples_per_strand,
                dtype=np.float32,
            )
            x = np.zeros_like(y)
            z = np.zeros_like(y)
            center_strand = np.column_stack((x, y, z)).astype(np.float32)
            lid_strands.append(
                _middle_top_local_to_world(
                    center_strand,
                    middle_segment,
                    middle_polyline,
                )
            )

    return lid_strands


def generate_full_spiral():
    length, radius, turns, positions = _default_spiral_parameters()

    # unit: nm
    base = SpiralBase(
        orientation_angle=0,
        radius=radius,
        turns=turns,
        height=length,
    )
    polyline = base.get_points(positions)
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    current_angle = float(base.orientation_angle + 2.0 * math.pi * base.turns)
    current_coords = polyline[-1]

    middle_scale = 1
    middle_segment = MiddleSegment(
        orientation_angle=current_angle,
        anchor_point=current_coords,
        turns=turns * middle_scale,
        height=length * middle_scale,
        start_x_radius=radius,
        end_x_radius_scale=3,
        y_radius=radius,
    )
    new_polyline = middle_segment.get_points(positions)
    assert new_polyline.ndim == 2 and new_polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    current_angle = float(
        middle_segment.orientation_angle + 2.0 * math.pi * middle_segment.turns
    )
    polyline = np.concatenate((polyline, new_polyline[1:]), axis=0)
    current_coords = polyline[-1]

    top_segment = TopSegment(
        orientation_angle=middle_segment.orientation_angle,
        anchor_point=current_coords,
        turns=turns * middle_scale,
        height=length * middle_scale,
        start_x_radius=radius * middle_segment.end_x_radius_scale,
        end_x_radius_scale=math.sqrt(3),
        y_radius=radius,
        phase_offset=_middle_segment_end_arc_angle(middle_segment),
    )
    new_polyline = top_segment.get_points(positions)
    assert new_polyline.ndim == 2 and new_polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    polyline = np.concatenate((polyline, new_polyline[1:]), axis=0)

    return polyline


def _pyvista_lines_from_segments(
    starts: Polyline,
    ends: Polyline,
    colors: npt.NDArray[np.uint8],
) -> pv.PolyData:
    assert starts.ndim == 2 and starts.shape[1] == 3, "starts must have shape (n, 3)"
    assert ends.shape == starts.shape, "ends must have the same shape as starts"
    assert colors.shape == (len(starts), 4), "colors must have shape (n, 4)"

    points = np.empty((2 * len(starts), 3), dtype=np.float32)
    points[0::2] = starts
    points[1::2] = ends
    line_indices = np.arange(2 * len(starts), dtype=np.int64).reshape((-1, 2))
    lines = np.column_stack((np.full(len(starts), 2, dtype=np.int64), line_indices))
    polydata = pv.PolyData(points)
    polydata.lines = lines.ravel()
    polydata.point_data["RGBA"] = np.repeat(colors, 2, axis=0)
    return polydata


def _lid_visualization_from_strands(
    lid_strands: list[Polyline],
    distance_between_base_pairs: float,
    dna_molecule_radius: float,
) -> pv.PolyData:
    assert len(lid_strands) > 0, "lid_strands must not be empty"
    lid_segment_starts: list[Polyline] = []
    lid_segment_ends: list[Polyline] = []
    lid_segment_colors: list[npt.NDArray[np.uint8]] = []

    for lid_strand in lid_strands:
        lid_base_pair_centers = reparametrize_arc_length(
            lid_strand,
            distance_between_base_pairs,
        )
        starts, ends, colors = dna_molecule_line_segments_from_base_pair_centers(
            base_pair_centers=lid_base_pair_centers,
            dna_molecule_radius=dna_molecule_radius,
        )
        colors[:] = np.array([255, 210, 64, 255], dtype=np.uint8)
        lid_segment_starts.append(starts)
        lid_segment_ends.append(ends)
        lid_segment_colors.append(colors)

    return _pyvista_lines_from_segments(
        np.concatenate(lid_segment_starts, axis=0),
        np.concatenate(lid_segment_ends, axis=0),
        np.concatenate(lid_segment_colors, axis=0),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    distance_between_base_pairs = 0.34
    dna_molecule_radius = 1.05
    dna_segment_radius = 0.05
    polyline = generate_full_spiral()
    lid_strands = generate_middle_top_lid_strands(
        dna_molecule_radius=dna_molecule_radius,
    )
    base_pair_centers = reparametrize_arc_length(
        polyline, distance_between_base_pairs
    )
    dna_segment_starts, dna_segment_ends, dna_segment_colors = (
        dna_molecule_line_segments_from_base_pair_centers(
            base_pair_centers=base_pair_centers,
            dna_molecule_radius=dna_molecule_radius,
        )
    )
    dna_visualization = _pyvista_lines_from_segments(
        dna_segment_starts,
        dna_segment_ends,
        dna_segment_colors,
    )
    lid_visualization = _lid_visualization_from_strands(
        lid_strands,
        distance_between_base_pairs,
        dna_molecule_radius,
    )

    if args.save:
        dna_mesh = dna_molecule_mesh_from_base_pair_centers(
            base_pair_centers=base_pair_centers,
            dna_molecule_radius=dna_molecule_radius,
            segment_radius=dna_segment_radius,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"full_double_spiral_{timestamp}"
        output_dir.mkdir(parents=True)
        np.save(output_dir / "polyline.npy", polyline)
        np.save(output_dir / "lid_strands.npy", np.stack(lid_strands, axis=0))
        np.save(output_dir / "base_pair_centers.npy", base_pair_centers)
        dna_mesh.export(output_dir / "full_double_spiral_dna.stl")
        dna_glb_path = output_dir / "full_double_spiral_dna.glb"
        dna_mesh.export(dna_glb_path)
        lightweight_tube = pv.lines_from_points(polyline).tube(
            radius=dna_molecule_radius
        )
        lightweight_tube.save(output_dir / "lightweight_tube.stl")
        lightweight_glb_path = save_lightweight_glb(
            polyline,
            output_dir / "lightweight_tube.glb",
            dna_molecule_radius,
        )
        print(f"Saved DNA GLB to {dna_glb_path} ({dna_glb_path.stat().st_size} bytes)")
        print(
            f"Saved lightweight GLB to {lightweight_glb_path} ({lightweight_glb_path.stat().st_size} bytes)"
        )
        print(output_dir)
        return 0

    plotter = pv.Plotter()
    plotter.add_mesh(  # type: ignore
        dna_visualization,
        scalars="RGBA",
        rgba=True,
        line_width=2,
        render_lines_as_tubes=True,
    )
    plotter.add_mesh(  # type: ignore
        lid_visualization,
        scalars="RGBA",
        rgba=True,
        line_width=4,
        render_lines_as_tubes=True,
    )
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
