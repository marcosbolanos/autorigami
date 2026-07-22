from __future__ import annotations

from dataclasses import dataclass

from autorigami.geometry.curvature import DEFAULT_EDGE_LENGTH_NM
from autorigami.geometry.reparametrize import (
    reparametrize_arc_length,
    validate_arc_length_sampling,
)
from autorigami.geometry.validation import (
    PolylineValidationResult,
    validate_polyline,
)
from autorigami.gravity.quasistatic import (
    QuasistaticGravityConfiguration,
    QuasistaticGravityResult,
    minimize_gravitational_energy,
)
from autorigami.types import Polyline


@dataclass(frozen=True)
class QuasistaticGravityWorkloadResult:
    material_points: Polyline
    points: Polyline
    optimization: QuasistaticGravityResult
    validation: PolylineValidationResult
    successful: bool
    message: str


def pack_under_gravity(
    polyline: Polyline,
    *,
    edge_length: float = DEFAULT_EDGE_LENGTH_NM,
    configuration: QuasistaticGravityConfiguration = (
        QuasistaticGravityConfiguration()
    ),
) -> QuasistaticGravityWorkloadResult:
    """Quasistatically lower, reparametrize, and standard-validate a chain."""
    assert edge_length > 0.0
    optimization = minimize_gravitational_energy(
        polyline,
        configuration=configuration,
    )
    points = reparametrize_arc_length(optimization.points, edge_length)
    validate_arc_length_sampling(points, interval=edge_length)
    validation = validate_polyline(
        points,
        maximum_angle=configuration.constraints.maximum_angle,
        minimum_distance=configuration.constraints.minimum_distance,
        valid_curvature_ignored_adjacent_edges=(
            configuration.constraints.ignored_adjacent_edges
        ),
        include_violation_masks=True,
    )
    successful = validation.valid
    message = (
        optimization.message
        if successful
        else (
            "reparametrized quasistatic result is invalid: "
            f"{validation.curvature_violation_count} curvature and "
            f"{validation.separation_violation_count} separation violations"
        )
    )
    return QuasistaticGravityWorkloadResult(
        material_points=optimization.points,
        points=points,
        optimization=optimization,
        validation=validation,
        successful=successful,
        message=message,
    )
