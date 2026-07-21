from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import cast

import numpy as np
import pyvista as pv

from autorigami.types import Polyline

TUBE_LENGTH_NM = 40.0
TUBE_RADIUS_NM = 16.0
BRANCH_ANGLE_RADIANS = math.radians(60.0)


def generate_y_centerlines(
    *,
    length: float = TUBE_LENGTH_NM,
) -> tuple[Polyline, Polyline, Polyline]:
    """Return three equal centerlines meeting at the origin as a planar Y."""
    assert length > 0.0, "length must be positive"
    junction = np.zeros(3, dtype=np.float32)
    horizontal_offset = np.float32(length * math.sin(BRANCH_ANGLE_RADIANS))
    vertical_offset = np.float32(length * math.cos(BRANCH_ANGLE_RADIANS))
    endpoints = (
        np.array([0.0, 0.0, -length], dtype=np.float32),
        np.array([-horizontal_offset, 0.0, vertical_offset], dtype=np.float32),
        np.array([horizontal_offset, 0.0, vertical_offset], dtype=np.float32),
    )
    return (
        np.stack((junction, endpoints[0]), axis=0),
        np.stack((junction, endpoints[1]), axis=0),
        np.stack((junction, endpoints[2]), axis=0),
    )


def tube_model_mesh(
    *,
    length: float = TUBE_LENGTH_NM,
    radius: float = TUBE_RADIUS_NM,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """Build the three equal-radius tube surfaces of the Y model."""
    assert radius > 0.0, "radius must be positive"
    centerlines = generate_y_centerlines(length=length)

    def tube(centerline: Polyline) -> pv.PolyData:
        return cast(
            pv.PolyData,
            pv.lines_from_points(centerline).tube(
                radius=radius,
                n_sides=48,
                capping=True,
            ),
        )

    return (
        tube(centerlines[0]),
        tube(centerlines[1]),
        tube(centerlines[2]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        type=Path,
        metavar="PNG",
        help="save a PNG instead of opening an interactive window",
    )
    args = parser.parse_args()

    plotter = pv.Plotter(
        off_screen=args.save is not None,
        window_size=[1000, 1000],
    )
    plotter.set_background("white")  # type: ignore
    for tube in tube_model_mesh():
        plotter.add_mesh(  # type: ignore
            tube,
            color="lightblue",
            smooth_shading=True,
        )
    plotter.view_xz(negative=True)  # type: ignore
    plotter.enable_parallel_projection()  # type: ignore
    plotter.camera.zoom(1.25)
    plotter.show(screenshot=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
