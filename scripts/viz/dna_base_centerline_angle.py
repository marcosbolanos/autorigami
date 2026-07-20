import argparse
import math
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from autorigami.types import Polyline

TURN_ANGLE_DEGREES = 3.25
TURN_ANGLE_RADIANS = math.radians(TURN_ANGLE_DEGREES)
BASE_LENGTH = 3.0
BASE_WIDTH = 0.65
BASE_DEPTH = 0.18
BASE_CENTER_DISTANCE = 3.0
CENTERLINE_LENGTH = 5.25
ANGLE_ARC_RADIUS = 1.1
ANGLE_ARC_POINT_COUNT = 80
LABEL_HEIGHT = 0.7

FIRST_BASE_COLOR = (79, 166, 94)
SECOND_BASE_COLOR = (139, 211, 146)
CENTERLINE_COLOR = (35, 105, 210)


def direction(angle: float) -> npt.NDArray[np.float32]:
    return np.array([math.cos(angle), math.sin(angle), 0.0], dtype=np.float32)


def base_mesh(
    center: npt.NDArray[np.float32],
    angle_degrees: float,
) -> pv.PolyData:
    assert center.shape == (3,), "center must have shape (3,)"

    mesh = pv.Cube(
        center=tuple(float(component) for component in center),
        x_length=BASE_LENGTH,
        y_length=BASE_WIDTH,
        z_length=BASE_DEPTH,
    )
    mesh.rotate_z(
        angle_degrees,
        point=tuple(float(component) for component in center),
        inplace=True,
    )
    return mesh


def angle_arc() -> Polyline:
    angles = np.linspace(
        0.0,
        TURN_ANGLE_RADIANS,
        ANGLE_ARC_POINT_COUNT,
        dtype=np.float32,
    )
    return ANGLE_ARC_RADIUS * np.column_stack(
        (
            np.cos(angles),
            np.sin(angles),
            np.zeros(ANGLE_ARC_POINT_COUNT, dtype=np.float32),
        )
    )


def render_angle_texture() -> npt.NDArray[np.uint8]:
    figure = Figure(figsize=(5.0, 1.7), dpi=200)
    figure.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(figure)
    color = tuple(float(channel) / 255.0 for channel in CENTERLINE_COLOR)
    figure.text(
        0.5,
        0.5,
        rf"$\theta = {TURN_ANGLE_DEGREES:.2f}^\circ$",
        color=color,
        fontsize=68,
        ha="center",
        va="center",
    )
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    occupied = np.argwhere(image[:, :, 3] > 0)
    assert occupied.size > 0, "angle texture must contain rendered text"
    y_min, x_min = occupied.min(axis=0)
    y_max, x_max = occupied.max(axis=0)
    padding = 10
    return image[
        max(int(y_min) - padding, 0) : min(int(y_max) + padding + 1, image.shape[0]),
        max(int(x_min) - padding, 0) : min(int(x_max) + padding + 1, image.shape[1]),
    ].copy()


def angle_label_mesh(label_texture: npt.NDArray[np.uint8]) -> pv.PolyData:
    assert label_texture.ndim == 3 and label_texture.shape[2] == 4, (
        "label_texture must have shape (height, width, 4)"
    )

    aspect_ratio = float(label_texture.shape[1]) / float(label_texture.shape[0])
    label_width = LABEL_HEIGHT * aspect_ratio
    center = np.array([2.2, 0.9, 0.12], dtype=np.float32)
    half_width = np.array([0.5 * label_width, 0.0, 0.0], dtype=np.float32)
    half_height = np.array([0.0, 0.5 * LABEL_HEIGHT, 0.0], dtype=np.float32)
    points = np.array(
        [
            center - half_width - half_height,
            center + half_width - half_height,
            center + half_width + half_height,
            center - half_width + half_height,
        ],
        dtype=np.float32,
    )
    label = pv.PolyData(points, np.array([4, 0, 1, 2, 3], dtype=np.int64))
    label.active_texture_coordinates = np.array(  # type: ignore
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    return label


def add_visualization(plotter: pv.Plotter) -> None:
    assert TURN_ANGLE_RADIANS > 0.0, "TURN_ANGLE_RADIANS must be positive"

    first_direction = direction(0.0)
    second_direction = direction(TURN_ANGLE_RADIANS)
    first_center = -BASE_CENTER_DISTANCE * first_direction
    second_center = BASE_CENTER_DISTANCE * second_direction
    plotter.add_mesh(  # type: ignore
        base_mesh(first_center, 0.0),
        color=FIRST_BASE_COLOR,
        smooth_shading=True,
    )
    plotter.add_mesh(  # type: ignore
        base_mesh(second_center, TURN_ANGLE_DEGREES),
        color=SECOND_BASE_COLOR,
        smooth_shading=True,
    )

    first_centerline = np.vstack(
        (-CENTERLINE_LENGTH * first_direction, 1.6 * first_direction)
    ).astype(np.float32)
    second_centerline = np.vstack(
        (np.zeros(3, dtype=np.float32), CENTERLINE_LENGTH * second_direction)
    )
    for centerline in (first_centerline, second_centerline):
        plotter.add_mesh(  # type: ignore
            pv.lines_from_points(centerline),
            color=CENTERLINE_COLOR,
            line_width=6,
            render_lines_as_tubes=True,
        )

    plotter.add_mesh(  # type: ignore
        pv.lines_from_points(angle_arc()),
        color=CENTERLINE_COLOR,
        line_width=6,
        render_lines_as_tubes=True,
    )
    label_texture = render_angle_texture()
    plotter.add_mesh(  # type: ignore
        angle_label_mesh(label_texture),
        texture=pv.Texture(label_texture),  # type: ignore
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    off_screen = args.save is not None
    plotter = pv.Plotter(off_screen=off_screen, window_size=[1400, 700])
    plotter.set_background("white")  # type: ignore
    add_visualization(plotter)
    plotter.view_xy()  # type: ignore
    plotter.camera.zoom(1.25)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(args.save)
        plotter.close()
        print(args.save)
        return 0

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
