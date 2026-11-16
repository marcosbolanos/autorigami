"""Build spiral centerlines from segments.

A `Spiral` starts empty. You give it a first segment, and each segment has
one or more termini: the open ends where the next segment can start.

    spiral = Spiral()
    stem = spiral.add_straight_segment(length=40.0, x_radius=16.0, y_radius=16.0)
    y = stem.termini[0].add_y_segment(
        middle_length=40.0,
        top_length=40.0,
        middle_opening_angle_deg=38.7,
        top_opening_angle_deg=23.7,
    )
    y.termini[0].add_straight_segment(length=20.0)
    y.termini[1].add_straight_segment(length=20.0)
    curves = spiral.curves()

All lengths are in nm and all segments wind upward along z. Each turn of a
segment is a horizontal ellipse, and the segment is the stack of these
ellipses. The curves are not repaired here: `Spiral.validate` reports if they
obey the curvature and separation limits of DNA, and
`autorigami.packing.pack_under_gravity` repairs one curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from autorigami.geometry.curvature import DEFAULT_EDGE_LENGTH_NM
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.spiral_generation.three_way_tube_v1 import MiddleSegment, TopSegment
from autorigami.spiral_generation.validation import (
    SpiralValidationResult,
    validate_curves,
)
from autorigami.types import Polyline

Vector2 = npt.NDArray[np.float64]

DEFAULT_RISE_PER_TURN: Final = 2.6
"""Height of one turn, in nm. It equals the crossover distance of DNA."""

# Samples of one turn before the arc-length resampling of `Spiral.curves`.
_SAMPLES_PER_TURN: Final = 2000


@dataclass(eq=False)
class Terminus:
    """An open end of a segment, where one more segment can start.

    Get a terminus from `Segment.termini`. Do not construct one yourself.
    Call one of the `add_..._segment` methods to continue the spiral from
    here. A terminus accepts one segment only.
    """

    spiral: Spiral = field(repr=False)
    center: Vector2
    """Horizontal center (x, y) of the last turn, in nm."""
    height: float
    """Height z at which the next segment starts, in nm."""
    x_radius: float
    """Radius of the last turn along its first axis, in nm."""
    y_radius: float
    """Radius of the last turn along its second axis, in nm."""
    axis_angle: float
    """Angle from the global x axis to the first axis of the turn, in radians."""
    phase: float
    """Angle of the next point on the turn, from its first axis, in radians."""
    drift: Vector2
    """Horizontal motion of the turn center for each nm of height.

    It is zero for a vertical tube. An arm of a Y segment leans outward, thus
    its terminus has a drift, and the segments after it lean the same way.
    """
    starts_new_curve: bool
    """True if a segment added here cannot continue the curve of its parent.

    A curve has one end only, thus the second arm of a Y segment starts a new
    curve. See `Spiral.curves`.
    """
    next_segment: Segment | None = None
    """The segment that starts here, or None if this terminus is still open."""

    def add_straight_segment(
        self,
        length: float,
        *,
        rise_per_turn: float = DEFAULT_RISE_PER_TURN,
    ) -> StraightSegment:
        """Continue with a tube that keeps the radii of this terminus.

        Args:
            length: Height of the segment along z, in nm.
            rise_per_turn: Height of one turn, in nm.

        Returns:
            The new segment. It has one terminus.
        """
        return self._add_tube(
            StraightSegment, length, self.x_radius, self.y_radius, rise_per_turn
        )

    def add_funnel_segment(
        self,
        length: float,
        *,
        end_x_radius: float,
        end_y_radius: float,
        rise_per_turn: float = DEFAULT_RISE_PER_TURN,
    ) -> FunnelSegment:
        """Continue with a tube whose radii change linearly with height.

        The radii start at the radii of this terminus.

        Args:
            length: Height of the segment along z, in nm.
            end_x_radius: Radius along the first axis at the top, in nm.
            end_y_radius: Radius along the second axis at the top, in nm.
            rise_per_turn: Height of one turn, in nm. For gravity packing,
                each radius must change by less than 2.6 nm in one turn.

        Returns:
            The new segment. It has one terminus.
        """
        return self._add_tube(
            FunnelSegment, length, end_x_radius, end_y_radius, rise_per_turn
        )

    def _add_tube[TubeSegment: FunnelSegment](
        self,
        segment_type: type[TubeSegment],
        length: float,
        end_x_radius: float,
        end_y_radius: float,
        rise_per_turn: float,
    ) -> TubeSegment:
        assert self.next_segment is None, "this terminus already has a segment"
        assert length > 0.0 and rise_per_turn > 0.0
        assert end_x_radius > 0.0 and end_y_radius > 0.0

        turns = length / rise_per_turn
        fraction = np.linspace(0.0, 1.0, _sample_count(turns))
        phase = self.phase + 2.0 * math.pi * turns * fraction
        x_radius = self.x_radius + (end_x_radius - self.x_radius) * fraction
        y_radius = self.y_radius + (end_y_radius - self.y_radius) * fraction
        start_height = self._start_height(rise_per_turn)
        center = self.center + np.outer(length * fraction, self.drift)
        local = np.column_stack((x_radius * np.cos(phase), y_radius * np.sin(phase)))
        points = np.column_stack(
            (
                center + local @ _rotation(self.axis_angle).T,
                start_height + length * fraction,
            )
        ).astype(np.float32)

        end = Terminus(
            spiral=self.spiral,
            center=center[-1],
            height=start_height + length,
            x_radius=end_x_radius,
            y_radius=end_y_radius,
            axis_angle=self.axis_angle,
            phase=float(phase[-1]),
            drift=self.drift,
            starts_new_curve=False,
        )
        segment = segment_type(parent=self, points=points, termini=(end,))
        self._attach(segment)
        return segment

    def add_y_segment(
        self,
        *,
        middle_length: float,
        top_length: float,
        middle_opening_angle_deg: float,
        top_opening_angle_deg: float,
        middle_rise_per_turn: float = DEFAULT_RISE_PER_TURN,
        top_rise_per_turn: float = 2.0 * DEFAULT_RISE_PER_TURN,
    ) -> YSegment:
        """Continue with a tube that divides into two arms.

        This is the shape of the three-way tube design 1. It has two parts.
        In the middle part, the round tube becomes wider along one axis and
        narrower at its waist. In the top part, the two ends become two round
        arms, each with the radius of this terminus. Each turn of the top
        part goes around the first arm, across the center, around the second
        arm, and back across the center.

        The arms open along the line from the tube axis to the point of this
        terminus. The Y segment is vertical: it ignores the drift of this
        terminus.

        The opening angle of a part is the angle between the vertical axis
        and the axis of an arm in that part. In one turn, each arm moves
        outward by `rise_per_turn * tan(opening_angle)`. For gravity packing,
        this distance must be less than 2.6 nm, or a turn cannot rest on the
        turn below it.

        Args:
            middle_length: Height of the middle part, in nm.
            top_length: Height of the top part, in nm.
            middle_opening_angle_deg: Opening angle of the middle part, in
                degrees, between 0 and 90. Design 1 uses 38.66.
            top_opening_angle_deg: Opening angle of the top part, in degrees,
                between 0 and 90. Design 1 uses 41.29 in its first version and
                23.71 in its version for gravity packing.
            middle_rise_per_turn: Height of one turn of the middle part, in nm.
            top_rise_per_turn: Height of one turn of the top part, in nm. The
                number of turns is adjusted by less than one half, so that
                the curve ends where the first arm is complete.

        Returns:
            The new segment. It has two termini, one for each arm.
            `termini[0]` continues the curve. `termini[1]` starts a new curve.
        """
        assert self.next_segment is None, "this terminus already has a segment"
        assert self.x_radius == self.y_radius, "a Y segment starts from a round tube"
        assert middle_length > 0.0 and top_length > 0.0
        assert middle_rise_per_turn > 0.0 and top_rise_per_turn > 0.0
        assert 0.0 < middle_opening_angle_deg < 90.0
        assert 0.0 < top_opening_angle_deg < 90.0

        radius = self.x_radius
        middle_slope = math.tan(math.radians(middle_opening_angle_deg))
        top_slope = math.tan(math.radians(top_opening_angle_deg))
        start_height = self._start_height(middle_rise_per_turn)
        orientation = self.axis_angle + self.phase
        start_point = np.array(
            [*(self.center + radius * _direction(orientation)), start_height],
            dtype=np.float32,
        )

        middle_turns = middle_length / middle_rise_per_turn
        middle_end_x_radius = radius + middle_length * middle_slope
        middle = MiddleSegment(
            orientation_angle=orientation,
            anchor_point=start_point,
            start_x_radius=radius,
            end_x_radius_scale=middle_end_x_radius / radius,
            y_radius=radius,
            turns=middle_turns,
            height=middle_length,
        )
        middle_points = middle.get_points(_positions(middle_turns))

        # The top part must end where the first arm is complete. In the local
        # angle of `TopSegment`, that is a multiple of one turn plus pi / 4.
        top_phase = 2.0 * math.pi * middle_turns
        nominal_turns = top_length / top_rise_per_turn
        remainder = ((0.25 * math.pi - top_phase) / (2.0 * math.pi)) % 1.0
        top_turns = max(round(nominal_turns - remainder), 1) + remainder
        top_end_x_radius = middle_end_x_radius + top_length * top_slope
        top = TopSegment(
            orientation_angle=orientation,
            anchor_point=middle_points[-1],
            start_x_radius=middle_end_x_radius,
            end_x_radius_scale=top_end_x_radius / middle_end_x_radius,
            y_radius=radius,
            turns=top_turns,
            height=top_length,
            phase_offset=top_phase,
        )
        top_points = top.get_points(_positions(top_turns))
        points = np.concatenate((middle_points, top_points[1:]), axis=0)

        # The second arm is complete one half of a turn before the end.
        outward = _direction(orientation)
        end_height = start_height + middle_length + top_length
        second_arm_fraction = 1.0 - 0.5 / top_turns
        second_arm_side = (
            middle_end_x_radius + second_arm_fraction * top_length * top_slope - radius
        )
        first_arm = Terminus(
            spiral=self.spiral,
            center=self.center + (top_end_x_radius - radius) * outward,
            height=end_height,
            x_radius=radius,
            y_radius=radius,
            axis_angle=orientation,
            phase=math.pi,
            drift=top_slope * outward,
            starts_new_curve=False,
        )
        second_arm = Terminus(
            spiral=self.spiral,
            center=self.center - second_arm_side * outward,
            height=end_height - top_length * 0.5 / top_turns,
            x_radius=radius,
            y_radius=radius,
            axis_angle=orientation,
            phase=0.0,
            drift=-top_slope * outward,
            starts_new_curve=True,
        )
        segment = YSegment(parent=self, points=points, termini=(first_arm, second_arm))
        self._attach(segment)
        return segment

    def _start_height(self, rise_per_turn: float) -> float:
        # A new curve starts one turn above the last turn of the arm.
        return self.height + (rise_per_turn if self.starts_new_curve else 0.0)

    def _attach(self, segment: Segment) -> None:
        self.next_segment = segment
        self.spiral.segments.append(segment)


@dataclass(eq=False)
class Segment:
    """One part of a spiral. Use the subclasses."""

    parent: Terminus = field(repr=False)
    """The terminus at which this segment starts."""
    points: Polyline = field(repr=False)
    """Dense samples of this segment, float32 with shape (n, 3), in nm."""
    termini: tuple[Terminus, ...]
    """The open ends of this segment. Add the next segments to these."""


@dataclass(eq=False)
class FunnelSegment(Segment):
    """A tube whose radii change linearly with height. It has one terminus."""

    termini: tuple[Terminus]


@dataclass(eq=False)
class StraightSegment(FunnelSegment):
    """A tube with constant radii. It has one terminus."""


@dataclass(eq=False)
class YSegment(Segment):
    """A tube that divides into two arms. It has two termini.

    `termini[0]` is the arm at which the curve ends: a segment added there
    continues the same curve. `termini[1]` is the other arm: a segment added
    there starts a new curve.
    """

    termini: tuple[Terminus, Terminus]


class Spiral:
    """A spiral centerline made of segments.

    Start with `add_straight_segment` or `add_funnel_segment`. Each call
    returns the new segment. Continue from `segment.termini[i]`, which has
    the same `add_..._segment` methods. `segments` lists all segments in the
    sequence in which you added them.
    """

    segments: list[Segment]
    """All segments, in the sequence in which they were added."""

    def __init__(self) -> None:
        self.segments = []

    def add_straight_segment(
        self,
        length: float,
        *,
        x_radius: float,
        y_radius: float,
        rise_per_turn: float = DEFAULT_RISE_PER_TURN,
    ) -> StraightSegment:
        """Start the spiral with a vertical tube of constant radii.

        The tube axis is the z axis and the first point is at height 0.

        Args:
            length: Height of the segment along z, in nm.
            x_radius: Radius along x, in nm.
            y_radius: Radius along y, in nm. Give the same value as
                `x_radius` for a round tube.
            rise_per_turn: Height of one turn, in nm.

        Returns:
            The new segment. It has one terminus.
        """
        return self._root(x_radius, y_radius).add_straight_segment(
            length, rise_per_turn=rise_per_turn
        )

    def add_funnel_segment(
        self,
        length: float,
        *,
        start_x_radius: float,
        start_y_radius: float,
        end_x_radius: float,
        end_y_radius: float,
        rise_per_turn: float = DEFAULT_RISE_PER_TURN,
    ) -> FunnelSegment:
        """Start the spiral with a vertical tube whose radii change linearly.

        The tube axis is the z axis and the first point is at height 0.

        Args:
            length: Height of the segment along z, in nm.
            start_x_radius: Radius along x at the bottom, in nm.
            start_y_radius: Radius along y at the bottom, in nm.
            end_x_radius: Radius along x at the top, in nm.
            end_y_radius: Radius along y at the top, in nm.
            rise_per_turn: Height of one turn, in nm. For gravity packing,
                each radius must change by less than 2.6 nm in one turn.

        Returns:
            The new segment. It has one terminus.
        """
        return self._root(start_x_radius, start_y_radius).add_funnel_segment(
            length,
            end_x_radius=end_x_radius,
            end_y_radius=end_y_radius,
            rise_per_turn=rise_per_turn,
        )

    def curves(self, interval: float = DEFAULT_EDGE_LENGTH_NM) -> list[Polyline]:
        """Return the centerlines of the spiral.

        A spiral without a Y segment is one curve. Each segment added to
        `termini[1]` of a Y segment starts one more curve, because one curve
        cannot follow two arms.

        Args:
            interval: Distance between consecutive points along each curve,
                in nm. The default is the distance between two base pairs.

        Returns:
            One float32 polyline with shape (n, 3) for each curve. The first
            is the curve that starts at the first segment.
        """
        assert self.segments, "the spiral has no segments"
        curves: list[Polyline] = []
        starts = [self.segments[0]]
        while starts:
            segment: Segment | None = starts.pop(0)
            parts: list[Polyline] = []
            while segment is not None:
                parts.append(segment.points if not parts else segment.points[1:])
                following: Segment | None = None
                for terminus in segment.termini:
                    if terminus.next_segment is None:
                        continue
                    if terminus.starts_new_curve:
                        starts.append(terminus.next_segment)
                    else:
                        following = terminus.next_segment
                segment = following
            curves.append(
                reparametrize_arc_length(np.concatenate(parts, axis=0), interval)
            )
        return curves

    def validate(self) -> SpiralValidationResult:
        """Examine the curvature and the separation of all curves.

        Each curve goes through `validate_polyline`, and each pair of curves
        is examined for edges that are too near. A Y segment has sharp
        corners before gravity packing, thus a spiral with a Y segment is
        usually not valid at this stage.

        Returns:
            The result. `result.valid` is True if there is no violation, and
            `print(result)` gives a summary.
        """
        return validate_curves(self.curves())

    def save(self, path: str | Path) -> None:
        """Write the curves of the spiral to a `.npz` file.

        Read them again with `load_curves`.

        Args:
            path: The file to write. Use the extension `.npz`.
        """
        np.savez(path, *self.curves())

    def _root(self, x_radius: float, y_radius: float) -> Terminus:
        assert not self.segments, (
            "the spiral already has a first segment: add to segment.termini"
        )
        assert x_radius > 0.0 and y_radius > 0.0
        return Terminus(
            spiral=self,
            center=np.zeros(2),
            height=0.0,
            x_radius=x_radius,
            y_radius=y_radius,
            axis_angle=0.0,
            phase=0.0,
            drift=np.zeros(2),
            starts_new_curve=False,
        )


def load_curves(path: str | Path) -> list[Polyline]:
    """Read the curves that `Spiral.save` wrote.

    Returns:
        One float32 polyline with shape (n, 3) for each curve, in the
        sequence of `Spiral.curves`.
    """
    with np.load(path) as archive:
        return [archive[f"arr_{index}"] for index in range(len(archive.files))]


def _sample_count(turns: float) -> int:
    return max(math.ceil(turns * _SAMPLES_PER_TURN), 2)


def _positions(turns: float) -> npt.NDArray[np.float32]:
    return np.linspace(0.0, 1.0, _sample_count(turns), dtype=np.float32)


def _direction(angle: float) -> Vector2:
    return np.array([math.cos(angle), math.sin(angle)])


def _rotation(angle: float) -> npt.NDArray[np.float64]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])
