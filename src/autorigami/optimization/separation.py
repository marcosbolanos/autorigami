from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt

from autorigami._native import (
    evaluate_tangent_point_hierarchical,
    segment_segment_distance,
)
from autorigami.geometry.curvature import (
    DEFAULT_EDGE_LENGTH_NM,
    DEFAULT_MAX_ANGLE_RADIANS,
    get_polyline_angles,
)
from autorigami.geometry.reparametrize import ArcLengthProjection
from autorigami.geometry.separation import (
    DEFAULT_MIN_DISTANCE,
    check_self_intersections,
    get_candidate_intersecting_edges,
)
from autorigami.optimization.fractional_sobolev import (
    FractionalSobolevPreconditioner,
)
from autorigami.types import Polyline

FloatArray = npt.NDArray[np.float64]
Termination = Literal["step_limit", "stationary", "constraint_blocked"]


class NativeTangentPointEvaluation(TypedDict):
    energy: float
    repulsive_energy: float
    attractive_energy: float
    differential: FloatArray
    exact_pair_count: int
    approximated_cluster_count: int


@dataclass(frozen=True)
class SeparationOptimizationSettings:
    """Numerical controls for adhesive separation optimization."""

    steps: int = 20
    maximum_vertex_step: float = 2e-2
    maximum_step_reductions: int = 12
    opening_angle: float = 0.5
    leaf_size: int = 8
    constraint_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        assert self.steps >= 0
        assert self.maximum_vertex_step > 0.0
        assert self.maximum_step_reductions >= 0
        assert self.opening_angle > 0.0
        assert self.leaf_size > 0
        assert self.constraint_tolerance > 0.0


@dataclass(frozen=True)
class SeparationOptimizationIteration:
    energy: float
    step_size: float
    step_reductions: int
    maximum_vertex_displacement: float
    maximum_edge_length_error: float
    maximum_angle: float
    minimum_separation: float


@dataclass(frozen=True)
class SeparationOptimizationResult:
    polyline: Polyline
    termination: Termination
    iterations: tuple[SeparationOptimizationIteration, ...]

    @property
    def converged(self) -> bool:
        """Whether descent stopped because its Sobolev direction vanished."""
        return self.termination == "stationary"


@dataclass(frozen=True)
class _OptimizationState:
    maximum_edge_length_error: float
    maximum_angle: float
    minimum_separation: float


def optimize_separation(
    polyline: Polyline,
    *,
    target_angle: float = DEFAULT_MAX_ANGLE_RADIANS,
    edge_length: float = DEFAULT_EDGE_LENGTH_NM,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    local_exclusion_length: float = 34.0,
    attraction_strength: float = 1.0,
    settings: SeparationOptimizationSettings = SeparationOptimizationSettings(),
) -> SeparationOptimizationResult:
    """Optimize adhesive contacts with validated fractional-Sobolev descent.

    Every proposed displacement is projected back to the original contour
    length, vertex count, barycenter, and arc-length parameterization.  A step
    is accepted only if curvature, separation, and parameterization all remain
    valid and the hierarchical tangent-point energy decreases.  When no such
    step is found, the last valid centerline is returned unchanged.
    """
    assert polyline.ndim == 2 and polyline.shape[1] == 3
    assert len(polyline) >= 4
    assert 0.0 < target_angle < np.pi
    assert edge_length > 0.0
    assert min_distance > 0.0
    assert local_exclusion_length > 0.0
    assert attraction_strength > 0.0

    points = np.asarray(polyline, dtype=np.float64).copy()
    ignored_adjacent_edges = int(np.ceil(local_exclusion_length / edge_length))
    parameterization = ArcLengthProjection.from_polyline(points, edge_length)
    _validate_state(
        points,
        parameterization=parameterization,
        target_angle=target_angle,
        min_distance=min_distance,
        ignored_adjacent_edges=ignored_adjacent_edges,
        tolerance=settings.constraint_tolerance,
    )

    diagnostics: list[SeparationOptimizationIteration] = []
    termination: Termination = "step_limit"
    for _ in range(settings.steps):
        evaluation = _evaluate_energy(
            points,
            min_distance=min_distance,
            attraction_strength=attraction_strength,
            local_exclusion_length=local_exclusion_length,
            settings=settings,
        )
        differential = np.asarray(evaluation["differential"], dtype=np.float64)
        preconditioner = FractionalSobolevPreconditioner(points, sigma=0.75)
        direction = -preconditioner.apply_inverse(differential)
        direction -= np.mean(direction, axis=0)
        largest_direction = float(np.max(np.linalg.norm(direction, axis=1)))
        if largest_direction <= 1e-12:
            termination = "stationary"
            break

        initial_step = min(
            1.0,
            settings.maximum_vertex_step / largest_direction,
        )
        accepted = _accept_step(
            points,
            direction=direction,
            initial_step=initial_step,
            initial_energy=float(evaluation["energy"]),
            directional_derivative=float(np.sum(differential * direction)),
            parameterization=parameterization,
            target_angle=target_angle,
            min_distance=min_distance,
            ignored_adjacent_edges=ignored_adjacent_edges,
            attraction_strength=attraction_strength,
            local_exclusion_length=local_exclusion_length,
            settings=settings,
        )
        if accepted is None:
            termination = "constraint_blocked"
            break

        points, energy, step_size, reductions = accepted
        state = _state(
            points,
            parameterization=parameterization,
            min_distance=min_distance,
            ignored_adjacent_edges=ignored_adjacent_edges,
        )
        diagnostics.append(
            SeparationOptimizationIteration(
                energy=energy,
                step_size=step_size,
                step_reductions=reductions,
                maximum_vertex_displacement=step_size * largest_direction,
                maximum_edge_length_error=state.maximum_edge_length_error,
                maximum_angle=state.maximum_angle,
                minimum_separation=state.minimum_separation,
            )
        )

    _validate_state(
        points,
        parameterization=parameterization,
        target_angle=target_angle,
        min_distance=min_distance,
        ignored_adjacent_edges=ignored_adjacent_edges,
        tolerance=settings.constraint_tolerance,
    )
    return SeparationOptimizationResult(
        polyline=np.asarray(points, dtype=np.float32),
        termination=termination,
        iterations=tuple(diagnostics),
    )


def _accept_step(
    points: FloatArray,
    *,
    direction: FloatArray,
    initial_step: float,
    initial_energy: float,
    directional_derivative: float,
    parameterization: ArcLengthProjection,
    target_angle: float,
    min_distance: float,
    ignored_adjacent_edges: int,
    attraction_strength: float,
    local_exclusion_length: float,
    settings: SeparationOptimizationSettings,
) -> tuple[FloatArray, float, float, int] | None:
    step_size = initial_step
    for reduction in range(settings.maximum_step_reductions + 1):
        candidate = parameterization.project(points + step_size * direction)
        try:
            _validate_state(
                candidate,
                parameterization=parameterization,
                target_angle=target_angle,
                min_distance=min_distance,
                ignored_adjacent_edges=ignored_adjacent_edges,
                tolerance=settings.constraint_tolerance,
            )
        except ValueError:
            step_size *= 0.5
            continue

        candidate_energy = float(
            _evaluate_energy(
                candidate,
                min_distance=min_distance,
                attraction_strength=attraction_strength,
                local_exclusion_length=local_exclusion_length,
                settings=settings,
            )["energy"]
        )
        armijo_bound = initial_energy + 1e-4 * step_size * directional_derivative
        if candidate_energy <= armijo_bound:
            return candidate, candidate_energy, step_size, reduction
        step_size *= 0.5
    return None


def _evaluate_energy(
    points: FloatArray,
    *,
    min_distance: float,
    attraction_strength: float,
    local_exclusion_length: float,
    settings: SeparationOptimizationSettings,
) -> NativeTangentPointEvaluation:
    return evaluate_tangent_point_hierarchical(
        np.asarray(points, dtype=np.float32),
        target_distance=min_distance,
        attraction_strength=attraction_strength,
        local_exclusion_length=local_exclusion_length,
        opening_angle=settings.opening_angle,
        leaf_size=settings.leaf_size,
    )


def _state(
    points: FloatArray,
    *,
    parameterization: ArcLengthProjection,
    min_distance: float,
    ignored_adjacent_edges: int,
) -> _OptimizationState:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    candidates = get_candidate_intersecting_edges(
        np.asarray(points, dtype=np.float32),
        min_distance=min_distance + 1.0,
        n_ignored_adjacent_edges=ignored_adjacent_edges,
    )
    distances = segment_segment_distance(
        np.asarray(points, dtype=np.float32),
        candidates,
    )
    return _OptimizationState(
        maximum_edge_length_error=float(
            np.max(
                np.abs(lengths - parameterization.sampling_interval),
                initial=0.0,
            )
        ),
        maximum_angle=float(
            np.max(
                get_polyline_angles(np.asarray(points, dtype=np.float32)),
                initial=0.0,
            )
        ),
        minimum_separation=min(
            (float(distance) for distance, _, _ in distances),
            default=np.inf,
        ),
    )


def _validate_state(
    points: FloatArray,
    *,
    parameterization: ArcLengthProjection,
    target_angle: float,
    min_distance: float,
    ignored_adjacent_edges: int,
    tolerance: float,
) -> None:
    state = _state(
        points,
        parameterization=parameterization,
        min_distance=min_distance,
        ignored_adjacent_edges=ignored_adjacent_edges,
    )
    separation = check_self_intersections(
        np.asarray(points, dtype=np.float32),
        min_euclid_distance=min_distance,
        n_ignored_adjacent_edges=ignored_adjacent_edges,
    )
    failures: list[str] = []
    try:
        parameterization.validate(points)
    except ValueError as error:
        failures.append(str(error))
    if state.maximum_angle > target_angle + tolerance:
        failures.append(f"maximum turning angle is {state.maximum_angle}")
    if separation["edges"] is not None:
        assert separation["distances"] is not None
        failures.append(
            f"{len(separation['edges'])} separation violations remain; "
            f"minimum distance is {min(separation['distances'])}"
        )
    if failures:
        raise ValueError(
            "invalid separation optimization state: " + "; ".join(failures)
        )
