from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from autorigami.geometry.validation import validate_polyline
from autorigami.gravity.constraints import (
    GravityConstraintConfiguration,
    linearize_gravity_constraints,
)
from autorigami.gravity.projection import (
    GravityProjectionConfiguration,
    GravityProjectionResult,
    project_gravity_constraints,
)
from autorigami.gravity.quadratic_step import (
    GravityQuadraticStepConfiguration,
    GravityQuadraticStepResult,
    solve_gravity_quadratic_step,
)
from autorigami.optimization.inextensibility import (
    maximum_edge_length_error,
    reference_edge_lengths,
)
from autorigami.types import Polyline

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class QuasistaticGravityConfiguration:
    constraints: GravityConstraintConfiguration = field(
        default_factory=GravityConstraintConfiguration
    )
    quadratic_step: GravityQuadraticStepConfiguration = field(
        default_factory=GravityQuadraticStepConfiguration
    )
    projection: GravityProjectionConfiguration = field(
        default_factory=GravityProjectionConfiguration
    )
    initial_floor_clearance: float = 0.001
    initial_trust_radius: float = 0.05
    maximum_trust_radius: float = 0.1
    minimum_trust_radius: float = 1e-5
    trust_shrink_factor: float = 0.5
    trust_growth_factor: float = 1.0
    maximum_iterations: int = 10
    minimum_energy_decrease: float = 1e-8
    displacement_tolerance: float = 1e-5
    relative_energy_tolerance: float = 1e-8
    stationary_iterations: int = 3
    material_edge_tolerance: float = 2e-4

    def __post_init__(self) -> None:
        assert self.initial_floor_clearance >= 0.0
        assert 0.0 < self.minimum_trust_radius <= self.initial_trust_radius
        assert self.initial_trust_radius <= self.maximum_trust_radius
        assert 0.0 < self.trust_shrink_factor < 1.0
        assert 1.0 <= self.trust_growth_factor <= 1.2
        assert self.maximum_iterations >= 1
        assert self.minimum_energy_decrease >= 0.0
        assert self.displacement_tolerance > 0.0
        assert self.relative_energy_tolerance > 0.0
        assert self.stationary_iterations >= 1
        assert self.material_edge_tolerance > 0.0


@dataclass(frozen=True)
class QuasistaticGravityIteration:
    iteration: int
    attempt: int
    accepted: bool
    reason: str
    energy_before: float
    energy_after: float
    trust_radius: float
    maximum_vertex_displacement: float
    height: float
    supported_vertex_count: int
    contact_count: int
    constraint_row_count: int
    matrix_nonzeros: int
    quadratic_step: GravityQuadraticStepResult
    projection: GravityProjectionResult | None


@dataclass(frozen=True)
class QuasistaticGravityResult:
    points: Polyline
    floor_height: float
    tube_radius: float
    converged: bool
    message: str
    initial_energy: float
    final_energy: float
    iterations: tuple[QuasistaticGravityIteration, ...]


def gravitational_energy(points: Polyline | FloatArray) -> float:
    """Return total gravitational potential for equal vertex weights."""
    values = np.asarray(points, dtype=np.float64)
    assert values.ndim == 2 and values.shape[1] == 3
    return float(np.sum(values[:, 2]))


def minimize_gravitational_energy(
    polyline: Polyline,
    *,
    configuration: QuasistaticGravityConfiguration = (
        QuasistaticGravityConfiguration()
    ),
) -> QuasistaticGravityResult:
    """Minimize gravity through global QP steps and nonlinear projection."""
    input_points = np.asarray(polyline, dtype=np.float32)
    assert input_points.ndim == 2 and input_points.shape[1] == 3
    assert len(input_points) >= 3
    assert np.all(np.isfinite(input_points))
    validation = validate_polyline(
        input_points,
        maximum_angle=configuration.constraints.maximum_angle,
        minimum_distance=configuration.constraints.minimum_distance,
        valid_curvature_ignored_adjacent_edges=(
            configuration.constraints.ignored_adjacent_edges
        ),
    )
    if not validation.valid:
        raise ValueError(
            "quasistatic gravity requires a green input; got "
            f"{validation.curvature_violation_count} curvature and "
            f"{validation.separation_violation_count} separation violations"
        )
    points = np.asarray(input_points, dtype=np.float64)
    lengths = reference_edge_lengths(points)
    tube_radius = 0.5 * configuration.constraints.minimum_distance
    floor_height = (
        float(np.min(points[:, 2]))
        - tube_radius
        - configuration.initial_floor_clearance
    )
    initial_energy = gravitational_energy(points)
    history: list[QuasistaticGravityIteration] = []
    trust_radius = configuration.initial_trust_radius
    stationary_count = 0
    converged = False
    message = "maximum quasistatic iterations reached"
    previous_displacement: FloatArray | None = None
    previous_pairs: tuple[tuple[int, int], ...] | None = None

    for iteration in range(configuration.maximum_iterations):
        accepted = False
        attempt = 0
        while trust_radius >= configuration.minimum_trust_radius:
            attempt += 1
            energy_before = gravitational_energy(points)
            linearization = linearize_gravity_constraints(
                points,
                lengths,
                floor_height=floor_height,
                tube_radius=tube_radius,
                trust_radius=trust_radius,
                configuration=configuration.constraints,
            )
            warm_start = (
                previous_displacement
                if previous_pairs == linearization.contact_pairs
                else None
            )
            quadratic = solve_gravity_quadratic_step(
                linearization,
                configuration=configuration.quadratic_step,
                warm_start=warm_start,
            )
            if not quadratic.succeeded:
                history.append(
                    _rejected_iteration(
                        iteration,
                        attempt,
                        f"QP status: {quadratic.status}",
                        points,
                        floor_height,
                        tube_radius,
                        trust_radius,
                        energy_before,
                        quadratic,
                        linearization.contact_count,
                        linearization.matrix.shape[0],
                        linearization.matrix.nnz,
                    )
                )
                trust_radius *= configuration.trust_shrink_factor
                continue
            assert quadratic.displacement is not None
            displacement = quadratic.displacement
            projection = project_gravity_constraints(
                points + displacement,
                lengths,
                floor_height=floor_height,
                tube_radius=tube_radius,
                constraint_configuration=configuration.constraints,
                projection_configuration=configuration.projection,
            )
            candidate = projection.points
            maximum_displacement = float(
                np.max(np.linalg.norm(candidate - points, axis=1), initial=0.0)
            )
            energy_after = gravitational_energy(candidate)
            candidate_validation = validate_polyline(
                np.asarray(candidate, dtype=np.float32),
                maximum_angle=configuration.constraints.maximum_angle,
                minimum_distance=configuration.constraints.minimum_distance,
                valid_curvature_ignored_adjacent_edges=(
                    configuration.constraints.ignored_adjacent_edges
                ),
            )
            edge_error = maximum_edge_length_error(candidate, lengths)
            energy_decrease = energy_before - energy_after
            reason = _candidate_rejection_reason(
                projection=projection,
                validation_valid=candidate_validation.valid,
                edge_error=edge_error,
                energy_decrease=energy_decrease,
                configuration=configuration,
            )
            history.append(
                QuasistaticGravityIteration(
                    iteration=iteration,
                    attempt=attempt,
                    accepted=reason is None,
                    reason=reason or "accepted",
                    energy_before=energy_before,
                    energy_after=energy_after,
                    trust_radius=trust_radius,
                    maximum_vertex_displacement=maximum_displacement,
                    height=float(np.ptp(candidate[:, 2])),
                    supported_vertex_count=int(
                        np.count_nonzero(
                            candidate[:, 2]
                            <= floor_height + tube_radius + 1e-5
                        )
                    ),
                    contact_count=linearization.contact_count,
                    constraint_row_count=linearization.matrix.shape[0],
                    matrix_nonzeros=linearization.matrix.nnz,
                    quadratic_step=quadratic,
                    projection=projection,
                )
            )
            if reason is not None:
                trust_radius *= configuration.trust_shrink_factor
                continue
            points = candidate
            previous_displacement = displacement
            previous_pairs = linearization.contact_pairs
            accepted = True
            relative_decrease = energy_decrease / max(1.0, abs(energy_before))
            if (
                maximum_displacement <= configuration.displacement_tolerance
                and relative_decrease <= configuration.relative_energy_tolerance
                and quadratic.primal_residual
                <= configuration.quadratic_step.absolute_tolerance
                and quadratic.dual_residual
                <= configuration.quadratic_step.absolute_tolerance
            ):
                stationary_count += 1
            else:
                stationary_count = 0
            trust_radius = min(
                configuration.maximum_trust_radius,
                configuration.trust_growth_factor * trust_radius,
            )
            break
        if not accepted:
            message = "trust region exhausted without a valid improving projection"
            break
        if stationary_count >= configuration.stationary_iterations:
            converged = True
            message = "quasistatic constrained stationarity reached"
            break

    result = np.asarray(points, dtype=np.float32)
    final_validation = validate_polyline(
        result,
        maximum_angle=configuration.constraints.maximum_angle,
        minimum_distance=configuration.constraints.minimum_distance,
        valid_curvature_ignored_adjacent_edges=(
            configuration.constraints.ignored_adjacent_edges
        ),
    )
    if not final_validation.valid:
        raise RuntimeError("quasistatic gravity returned a non-green material chain")
    return QuasistaticGravityResult(
        points=result,
        floor_height=floor_height,
        tube_radius=tube_radius,
        converged=converged,
        message=message,
        initial_energy=initial_energy,
        final_energy=gravitational_energy(points),
        iterations=tuple(history),
    )


def _candidate_rejection_reason(
    *,
    projection: GravityProjectionResult,
    validation_valid: bool,
    edge_error: float,
    energy_decrease: float,
    configuration: QuasistaticGravityConfiguration,
) -> str | None:
    if not projection.converged:
        return projection.message
    if not validation_valid:
        return "standard validator rejected projected candidate"
    if edge_error > configuration.material_edge_tolerance:
        return "projected material-edge error exceeds tolerance"
    if energy_decrease <= configuration.minimum_energy_decrease:
        return "projected candidate did not lower gravitational energy"
    return None


def _rejected_iteration(
    iteration: int,
    attempt: int,
    reason: str,
    points: FloatArray,
    floor_height: float,
    tube_radius: float,
    trust_radius: float,
    energy: float,
    quadratic: GravityQuadraticStepResult,
    contact_count: int,
    constraint_row_count: int,
    matrix_nonzeros: int,
) -> QuasistaticGravityIteration:
    return QuasistaticGravityIteration(
        iteration=iteration,
        attempt=attempt,
        accepted=False,
        reason=reason,
        energy_before=energy,
        energy_after=energy,
        trust_radius=trust_radius,
        maximum_vertex_displacement=0.0,
        height=float(np.ptp(points[:, 2])),
        supported_vertex_count=int(
            np.count_nonzero(points[:, 2] <= floor_height + tube_radius + 1e-5)
        ),
        contact_count=contact_count,
        constraint_row_count=constraint_row_count,
        matrix_nonzeros=matrix_nonzeros,
        quadratic_step=quadratic,
        projection=None,
    )
