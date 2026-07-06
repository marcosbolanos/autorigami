import math
import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod

from jaxtyping import Float32
import pyvista as pv
import numpy as np

type Vector3 = Float32[np.ndarray, "3"]
type Polyline = Float32[np.ndarray, "n 3"]


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

# Middle of the spiral, funnel-shaped
@dataclass
class SpiralMiddleFunnel(SpiralObject):
    starting_angle: float
    starting_coords: Vector3
    starting_x_radius: float
    x_radius_increase_factor: float
    y_radius: float
    winding_frequency: float
    length: float

    def get_point(self, position: float) -> Vector3:
        assert 0 <= position <= 1, "position must be between 0 and 1"

        angle = self.starting_angle + 2.0 * math.pi * self.winding_frequency * position
        local_angle = angle - self.starting_angle
        # The key change is that we increasingly distort the x radius
        x_radius = self.starting_x_radius * (1.0 + position * (self.x_radius_increase_factor - 1.0))
        inward_position = max(0.0, (position - 0.2) / 0.8)
        center = np.array([0.0, 0.0, self.starting_coords[2]], dtype=np.float32)
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
    middle_funnel = SpiralMiddleFunnel(
        starting_angle=current_angle,
        starting_coords=current_coords,
        winding_frequency=winding_frequency * middle_scale,
        length=length * middle_scale,
        starting_x_radius=radius,
        x_radius_increase_factor=3,
        y_radius=radius
    )
    new_polyline, current_angle = middle_funnel.discretize(10000)
    polyline = np.concatenate((polyline, new_polyline[1:]), axis=0)
    
    return polyline


def main() -> int:
    dna_molecule_radius = 1.05
    polyline = generate_full_spiral()
    tube = pv.lines_from_points(polyline).tube(radius=dna_molecule_radius)
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
