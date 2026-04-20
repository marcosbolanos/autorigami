#pragma once

#include <cstddef>
#include <vector>

#include "autorigami/vec3.h"

namespace autorigami {

struct PiecewiseHermiteData {
    std::vector<Vec3> points;
    std::vector<Vec3> tangents;
};

struct PiecewiseHermiteGeneratorRunData {
    std::size_t point_count;
    std::size_t segment_count;
    double parameter_step;
};

struct PiecewiseHermiteGeneratorResult {
    PiecewiseHermiteData piecewise_hermite;
    PiecewiseHermiteGeneratorRunData run_data;
};

[[nodiscard]] PiecewiseHermiteGeneratorResult piecewise_hermite_generator();

}  // namespace autorigami
