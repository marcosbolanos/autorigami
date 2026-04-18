#pragma once

#include "autorigami/splines.h"

namespace autorigami {

struct CurvatureMaxResult {
    double lower_bound;
    double upper_bound;
    double best_t;
};

struct CurvatureMaxSolverConfig {
    double abs_tolerance_squared = 1e-10;
    int max_depth = 40;
    double min_interval_width = 1e-12;
};

[[nodiscard]] CurvatureMaxResult max_curvature_of_segment(
    const CubicPowerBasisSegment& segment,
    const CurvatureMaxSolverConfig& config = {}
);

}  // namespace autorigami
