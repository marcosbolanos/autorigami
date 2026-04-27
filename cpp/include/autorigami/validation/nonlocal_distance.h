#pragma once

#include <vector>

#include "autorigami/vec3.h"

namespace autorigami {

struct NonlocalDistanceValidationResult {
    int violation_count;
    double minimum_checked_distance;
};

[[nodiscard]] NonlocalDistanceValidationResult validate_polyline_nonlocal_distance(
    const std::vector<Vec3>& points,
    double minimum_separation,
    double nonlocal_window,
    bool stop_on_first_violation = false
);

}  // namespace autorigami
