"""Interactive 3D views for notebooks.

Each function returns a plotly figure. A notebook shows the figure when it is
the last expression of a cell. Drag to rotate, scroll to zoom.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import plotly.graph_objects as go

from autorigami.spiral_generation.spiral import Segment, Spiral
from autorigami.types import Polyline

_COLORS: Final = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#EECA3B",
    "#9D755D",
)
_FADED_COLOR: Final = "#C8C8C8"
# The segments have 2000 samples in one turn. The view shows 1 of each 20.
_SEGMENT_STRIDE: Final = 20


def show_spiral(spiral: Spiral, *, highlight: Segment | None = None) -> go.Figure:
    """Show a spiral in 3D, with one line for each segment.

    Args:
        spiral: The spiral to show.
        highlight: If given, this segment has a color and all other segments
            are gray. Use it to show the segment that a cell added.

    Returns:
        The figure. The legend gives the index of each segment in
        `spiral.segments` and its type.
    """
    figure = _empty_figure()
    for index, segment in enumerate(spiral.segments):
        points = segment.points[::_SEGMENT_STRIDE]
        faded = highlight is not None and segment is not highlight
        figure.add_trace(
            _line(
                points,
                name=f"{index}: {type(segment).__name__}",
                color=_FADED_COLOR if faded else _COLORS[index % len(_COLORS)],
            )
        )
    return figure


def show_curves(curves: Sequence[Polyline]) -> go.Figure:
    """Show centerlines in 3D, with one line for each curve.

    Args:
        curves: Polylines with shape (n, 3), in nm, for example from
            `Spiral.curves` or `load_curves`.

    Returns:
        The figure.
    """
    figure = _empty_figure()
    for index, curve in enumerate(curves):
        figure.add_trace(
            _line(curve, name=f"curve {index}", color=_COLORS[index % len(_COLORS)])
        )
    return figure


def _line(points: Polyline, *, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        name=name,
        line={"color": color, "width": 4},
        hoverinfo="name",
    )


def _empty_figure() -> go.Figure:
    axis = {"title": {"text": ""}, "showbackground": False}
    return go.Figure(
        layout=go.Layout(
            scene={
                "aspectmode": "data",
                "xaxis": {**axis, "title": {"text": "x (nm)"}},
                "yaxis": {**axis, "title": {"text": "y (nm)"}},
                "zaxis": {**axis, "title": {"text": "z (nm)"}},
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            height=520,
            legend={"x": 0.0, "y": 1.0},
        )
    )
