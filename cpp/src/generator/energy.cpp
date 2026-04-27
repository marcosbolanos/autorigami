#include "autorigami/generator/energy.h"

#include <cmath>
#include <limits>

#include "autorigami/curvature_math.h"

namespace autorigami {

namespace {

[[nodiscard]] Vec3 to_vec3(const geometrycentral::Vector3& value) {
    return {.x = value.x, .y = value.y, .z = value.z};
}

}  // namespace

double evaluate_end_of_spline_energy(
    const PiecewiseHermiteData& spline,
    double max_curvature,
    double curvature_tolerance
) {
    if (spline.points.size() < 2 || spline.tangents.size() < 2) {
        return 0.0;
    }
    if (spline.points.size() != spline.tangents.size()) {
        return std::numeric_limits<double>::infinity();
    }

    const std::size_t segment_count = spline.points.size() - 1;
    const std::size_t first_segment = segment_count > 1 ? segment_count - 2 : 0;
    constexpr double barrier_epsilon = 1e-12;
    double energy = 0.0;

    for (std::size_t segment_index = first_segment; segment_index < segment_count; ++segment_index) {
        const CubicHermiteSegment segment{
            .p0 = to_vec3(spline.points[segment_index]),
            .p1 = to_vec3(spline.points[segment_index + 1]),
            .m0 = to_vec3(spline.tangents[segment_index]),
            .m1 = to_vec3(spline.tangents[segment_index + 1]),
        };
        const CurvatureMaxResult curvature = max_curvature_of_segment(segment.to_power_basis());
        const double slack = (max_curvature + curvature_tolerance) - curvature.upper_bound;
        if (slack <= barrier_epsilon) {
            return std::numeric_limits<double>::infinity();
        }
        energy += -std::log(slack);
    }

    return energy;
}

}  // namespace autorigami
