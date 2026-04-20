#pragma once

#include "autorigami/generator.h"

namespace autorigami {

[[nodiscard]] PiecewiseHermiteData bootstrap_from_seed(
    const geometrycentral::Vector3& seed,
    const GeneratorAxis& axis
);

[[nodiscard]] PiecewiseHermiteData optimize_end_of_spline(
    const PiecewiseHermiteData& current,
    const GeneratorAxis& axis,
    double max_curvature,
    double curvature_tolerance
);

}  // namespace autorigami
