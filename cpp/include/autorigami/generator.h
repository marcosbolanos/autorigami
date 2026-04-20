#pragma once

#include <vector>

#include "autorigami/vec3.h"

namespace autorigami {

struct PiecewiseHermiteData {
    std::vector<Vec3> points;
    std::vector<Vec3> tangents;
};

[[nodiscard]] PiecewiseHermiteData piecewise_hermite_generator();

}  // namespace autorigami
