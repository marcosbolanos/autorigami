#pragma once

#include <vector>

#include "autorigami/curvature_math.h"
#include "autorigami/splines.h"
#include "autorigami/vec3.h"

namespace autorigami {

struct ConstraintReport {
    int compliant_count;
    int total_count;

    [[nodiscard]] double ratio() const;
    [[nodiscard]] bool operator==(const ConstraintReport& other) const = default;
};

struct ValidationReport {
    ConstraintReport separation;
    ConstraintReport curvature;

    [[nodiscard]] bool operator==(const ValidationReport& other) const = default;
};

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

[[nodiscard]] ValidationReport validate_polyline_constraints(
    const std::vector<Vec3>& points,
    double separation,
    double max_curvature,
    int neighbor_exclusion
);

}  // namespace autorigami
