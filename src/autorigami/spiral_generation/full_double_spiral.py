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

        # Instantiate a MiddleSegment, we'll interpolate point coordinates towards the new geometry
        middle_segment = MiddleSegment(
            orientation_angle=self.orientation_angle,
            anchor_point=self.anchor_point,
            start_x_radius=self.start_x_radius,
            end_x_radius_scale=self.end_x_radius_scale,
            y_radius=self.y_radius,
            turns=self.turns,
            height=self.height,
            phase_offset=self.phase_offset,
            inward_position_offset=1.0,
        )
        middle_polyline = middle_segment.get_points(positions)
        assert middle_polyline.ndim == 2 and middle_polyline.shape[1] == 3, (
            "middle_polyline must have shape (n, 3)"
        )

        local_angle = self.phase_offset + 2.0 * math.pi * self.turns * positions
        x_radius = self.start_x_radius * (
            1.0 + positions * (self.end_x_radius_scale - 1.0)
        )
        q = np.mod(local_angle + 0.25 * math.pi, 2.0 * math.pi)
        side_center_x = x_radius - self.y_radius
        geometry_position = positions * positions * (3.0 - 2.0 * positions)
        join_angle = 0.5 * math.pi + 0.5 * math.pi * geometry_position
        bridge_y_scale = 0.65
        x = np.empty_like(positions)
        y = np.empty_like(positions)

        mask = q < 0.5 * math.pi
        q_m = q[mask]
        side_m = side_center_x[mask]
        join_m = join_angle[mask]
        angle = -join_m + (q_m / (0.5 * math.pi)) * 2.0 * join_m
        x[mask] = side_m + self.y_radius * np.cos(angle)
        y[mask] = self.y_radius * np.sin(angle)

        mask = (0.5 * math.pi <= q) & (q < math.pi)
        if np.any(mask):
            side_m = side_center_x[mask]
            join_m = join_angle[mask]
            u = (q[mask] - 0.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            branch_x = side_m * (1.0 - 2.0 * u)
            branch_y = self.y_radius * (1.0 - 0.95 * np.sin(math.pi * u) ** 2)
            bridge_mask = ~np.isclose(np.cos(join_m), 0.0, atol=1e-6)
            if np.any(bridge_mask):
                bridge_join = join_m[bridge_mask]
                bridge_start_x = side_m[bridge_mask] + self.y_radius * np.cos(
                    bridge_join
                )
                bridge_start_y = self.y_radius * np.sin(bridge_join)
                bridge_start_angle = np.arctan(bridge_y_scale * np.tan(bridge_join))
                bridge_x_radius = bridge_start_x / np.cos(bridge_start_angle)
                bridge_y_radius = bridge_y_scale * bridge_x_radius
                bridge_center_y = bridge_start_y - bridge_y_radius * np.sin(
                    bridge_start_angle
                )
                bridge_angle = bridge_start_angle + u[bridge_mask] * (
                    -math.pi - 2.0 * bridge_start_angle
                )
                branch_x[bridge_mask] = bridge_x_radius * np.cos(bridge_angle)
                branch_y[bridge_mask] = bridge_center_y + bridge_y_radius * np.sin(
                    bridge_angle
                )
            x[mask] = branch_x
            y[mask] = branch_y

        mask = (math.pi <= q) & (q < 1.5 * math.pi)
        if np.any(mask):
            q_m = q[mask]
            side_m = side_center_x[mask]
            join_m = join_angle[mask]
            new_angle = (
                math.pi - join_m + ((q_m - math.pi) / (0.5 * math.pi)) * 2.0 * join_m
            )
            x[mask] = -side_m + self.y_radius * np.cos(new_angle)
            y[mask] = self.y_radius * np.sin(new_angle)

        mask = 1.5 * math.pi <= q
        if np.any(mask):
            side_m = side_center_x[mask]
            join_m = join_angle[mask]
            u = (q[mask] - 1.5 * math.pi) / (0.5 * math.pi)
            u = u * u * (3.0 - 2.0 * u)
            branch_x = side_m * (-1.0 + 2.0 * u)
            branch_y = -self.y_radius * (1.0 - 0.95 * np.sin(math.pi * u) ** 2)
            bridge_mask = ~np.isclose(np.cos(join_m), 0.0, atol=1e-6)
            if np.any(bridge_mask):
                bridge_join = join_m[bridge_mask]
                bridge_start_x = side_m[bridge_mask] + self.y_radius * np.cos(
                    bridge_join
                )
                bridge_start_y = self.y_radius * np.sin(bridge_join)
                bridge_start_angle = np.arctan(bridge_y_scale * np.tan(bridge_join))
                bridge_x_radius = bridge_start_x / np.cos(bridge_start_angle)
                bridge_y_radius = bridge_y_scale * bridge_x_radius
                bridge_center_y = -bridge_start_y + bridge_y_radius * np.sin(
                    bridge_start_angle
                )
                bridge_angle = (
                    math.pi
                    + bridge_start_angle
                    + u[bridge_mask] * (-math.pi - 2.0 * bridge_start_angle)
                )
                branch_x[bridge_mask] = bridge_x_radius * np.cos(bridge_angle)
                branch_y[bridge_mask] = bridge_center_y + bridge_y_radius * np.sin(
                    bridge_angle
                )
            x[mask] = branch_x
            y[mask] = branch_y

        cos_orientation = math.cos(self.orientation_angle)
        sin_orientation = math.sin(self.orientation_angle)
        x, y = (
            x * cos_orientation - y * sin_orientation,
            x * sin_orientation + y * cos_orientation,
        )

        start_q = (self.phase_offset + 0.25 * math.pi) % (2.0 * math.pi)
        start_side_center_x = self.start_x_radius - self.y_radius
        start_join_angle = 0.5 * math.pi
        if start_q < 0.5 * math.pi:
            start_arc_angle = (
                -start_join_angle + (start_q / (0.5 * math.pi)) * 2.0 * start_join_angle
            )
            start_x = start_side_center_x + self.y_radius * math.cos(start_arc_angle)
            start_y = self.y_radius * math.sin(start_arc_angle)
        elif start_q < math.pi:
            start_u = (start_q - 0.5 * math.pi) / (0.5 * math.pi)
            start_u = start_u * start_u * (3.0 - 2.0 * start_u)
            start_x = start_side_center_x * (1.0 - 2.0 * start_u)
            start_y = self.y_radius * (1.0 - 0.95 * math.sin(math.pi * start_u) ** 2)
        elif start_q < 1.5 * math.pi:
            start_arc_angle = (
                math.pi
                - start_join_angle
                + ((start_q - math.pi) / (0.5 * math.pi)) * 2.0 * start_join_angle
            )
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
        top_polyline: Polyline = center + np.column_stack((x, y, z)).astype(np.float32)
        polyline: Polyline = (
            middle_polyline * (1.0 - geometry_position[:, None])
            + top_polyline * geometry_position[:, None]
        ).astype(np.float32)
        return polyline[0] if scalar_input else polyline


def generate_full_spiral():
    nm_per_full_turn = 2.6
    length = 40
    radius = 16
    turns = length / nm_per_full_turn
    positions = np.linspace(0.0, 1.0, 10000, dtype=np.float32)

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
        turns=0.5 * turns * middle_scale,
        height=length * middle_scale,
        start_x_radius=radius * middle_segment.end_x_radius_scale,
        end_x_radius_scale=math.sqrt(3),
        y_radius=radius,
        phase_offset=current_angle - middle_segment.orientation_angle,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    distance_between_base_pairs = 0.34
    dna_molecule_radius = 1.05
    dna_segment_radius = 0.05
    polyline = generate_full_spiral()
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
