#pragma once

#include <vector>

#include "autorigami/validation/nonlocal_distance.h"
#include "autorigami/vec3.h"

namespace autorigami::acap {

struct ToolpathStats {
    int point_count;
    double length_world;
    NonlocalDistanceValidationResult nonlocal_distance;
};

[[nodiscard]] ToolpathStats compute_toolpath_stats(
    const std::vector<Vec3>& points,
    double minimum_separation_world,
    double nonlocal_window_world
);

}  // namespace autorigami::acap
