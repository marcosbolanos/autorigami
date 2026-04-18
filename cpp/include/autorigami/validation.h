#pragma once

#include "autorigami/curvature_math.h"
#include "autorigami/splines.h"

namespace autorigami {

[[nodiscard]] bool validate_curve_curvature(
    const CubicPowerBasisSegment& segment,
    double max_curvature,
    double curvature_tolerance,
    const CurvatureMaxSolverConfig& solver_config = {}
);

[[nodiscard]] bool validate_curve_curvature(
    const CubicHermiteSegment& segment,
    double max_curvature,
    double curvature_tolerance,
    const CurvatureMaxSolverConfig& solver_config = {}
);

[[nodiscard]] bool validate_piecewise_curve_curvature(
    const PiecewiseCubicHermiteSpline& spline,
    double max_curvature,
    double curvature_tolerance,
    const CurvatureMaxSolverConfig& solver_config = {}
);

}  // namespace autorigami
