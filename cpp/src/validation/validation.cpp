#include "autorigami/validation.h"

#include <stdexcept>

namespace autorigami {

bool validate_curve_curvature(
    const CubicPowerBasisSegment& segment,
    double max_curvature,
    double curvature_tolerance,
    const CurvatureMaxSolverConfig& solver_config
) {
    if (max_curvature < 0.0) {
        throw std::invalid_argument("max_curvature must be >= 0");
    }
    if (curvature_tolerance < 0.0) {
        throw std::invalid_argument("curvature_tolerance must be >= 0");
    }

    const CurvatureMaxResult result = max_curvature_of_segment(segment, solver_config);
    return result.upper_bound <= max_curvature + curvature_tolerance;
}

bool validate_curve_curvature(
    const CubicHermiteSegment& segment,
    double max_curvature,
    double curvature_tolerance,
    const CurvatureMaxSolverConfig& solver_config
) {
    return validate_curve_curvature(
        segment.to_power_basis(),
        max_curvature,
        curvature_tolerance,
        solver_config
    );
}

bool validate_piecewise_curve_curvature(
    const PiecewiseCubicHermiteSpline& spline,
    double max_curvature,
    double curvature_tolerance,
    const CurvatureMaxSolverConfig& solver_config
) {
    if (spline.empty()) {
        throw std::invalid_argument("spline must contain at least one segment");
    }

    for (const CubicHermiteSegment& segment : spline.segments()) {
        if (!validate_curve_curvature(
                segment,
                max_curvature,
                curvature_tolerance,
                solver_config
            )) {
            return false;
        }
    }

    return true;
}

}  // namespace autorigami
