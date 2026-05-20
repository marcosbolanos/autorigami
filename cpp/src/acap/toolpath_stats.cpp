#include "autorigami/acap/toolpath_stats.h"

#include <cstddef>
#include <stdexcept>

namespace autorigami::acap {

namespace {

[[nodiscard]] Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
}

[[nodiscard]] double polyline_length(const std::vector<Vec3>& points) {
    double length = 0.0;
    for (std::size_t index = 1; index < points.size(); ++index) {
        length += norm(subtract(points[index], points[index - 1]));
    }
    return length;
}

}  // namespace

ToolpathStats compute_toolpath_stats(
    const std::vector<Vec3>& points,
    double minimum_separation_world,
    double nonlocal_window_world
) {
    if (points.size() < 2) {
        throw std::invalid_argument("ACAP toolpath requires at least two points");
    }

    return ToolpathStats{
        .point_count = static_cast<int>(points.size()),
        .length_world = polyline_length(points),
        .nonlocal_distance = validate_polyline_nonlocal_distance(
            points,
            minimum_separation_world,
            nonlocal_window_world,
            false
        ),
    };
}

}  // namespace autorigami::acap
