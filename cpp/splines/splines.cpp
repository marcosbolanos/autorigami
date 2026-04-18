#include "autorigami/splines.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace autorigami {

namespace {

[[nodiscard]] std::vector<double> compute_centripetal_parameters(const std::vector<Vec3>& points) {
    constexpr double alpha = 0.5;

    std::vector<double> parameters(points.size(), 0.0);
    for (std::size_t index = 1; index < points.size(); ++index) {
        const double chord_length = autorigami::norm(points[index] - points[index - 1]);
        parameters[index] = parameters[index - 1] + std::pow(chord_length, alpha);
    }
    return parameters;
}

[[nodiscard]] std::vector<Vec3> compute_point_tangents(
    const std::vector<Vec3>& points,
    const std::vector<double>& parameters
) {
    std::vector<Vec3> tangents(points.size(), {0.0, 0.0, 0.0});

    tangents.front() = (points[1] - points[0]) / (parameters[1] - parameters[0]);
    tangents.back() = (points[points.size() - 1] - points[points.size() - 2]) /
                      (parameters[parameters.size() - 1] - parameters[parameters.size() - 2]);

    for (std::size_t index = 1; index + 1 < points.size(); ++index) {
        const double dt_prev = parameters[index] - parameters[index - 1];
        const double dt_next = parameters[index + 1] - parameters[index];
        const Vec3 secant_prev = (points[index] - points[index - 1]) / dt_prev;
        const Vec3 secant_next = (points[index + 1] - points[index]) / dt_next;
        tangents[index] = (secant_prev * dt_next + secant_next * dt_prev) / (dt_prev + dt_next);
    }

    return tangents;
}

[[nodiscard]] PiecewiseCubicHermiteSpline build_catmull_rom_centripetal_spline(
    const std::vector<Vec3>& points
) {
    const std::vector<double> parameters = compute_centripetal_parameters(points);
    const std::vector<Vec3> point_tangents = compute_point_tangents(points, parameters);

    std::vector<CubicHermiteSegment> segments;
    segments.reserve(points.size() - 1);

    for (std::size_t index = 0; index + 1 < points.size(); ++index) {
        const double dt = parameters[index + 1] - parameters[index];
        segments.push_back(CubicHermiteSegment{
            .p0 = points[index],
            .p1 = points[index + 1],
            .m0 = point_tangents[index] * dt,
            .m1 = point_tangents[index + 1] * dt,
        });
    }

    return PiecewiseCubicHermiteSpline(std::move(segments));
}

}  // namespace

CubicPowerBasisSegment CubicHermiteSegment::to_power_basis() const {
    return CubicPowerBasisSegment{
        .a = 2.0 * p0 - 2.0 * p1 + m0 + m1,
        .b = -3.0 * p0 + 3.0 * p1 - 2.0 * m0 - m1,
        .c = m0,
        .d = p0,
    };
}

PiecewiseCubicHermiteSpline::PiecewiseCubicHermiteSpline(
    std::vector<CubicHermiteSegment> segments
)
    : segments_(std::move(segments)) {}

PiecewiseCubicHermiteSpline PiecewiseCubicHermiteSpline::from_polyline(
    const std::vector<Vec3>& points,
    TangentPolicy tangent_policy
) {
    if (points.size() < 2) {
        throw std::invalid_argument("need at least 2 points");
    }

    switch (tangent_policy) {
        case TangentPolicy::CatmullRomCentripetal:
            return build_catmull_rom_centripetal_spline(points);
    }

    throw std::invalid_argument("unsupported tangent policy");
}

const std::vector<CubicHermiteSegment>& PiecewiseCubicHermiteSpline::segments() const {
    return segments_;
}

std::size_t PiecewiseCubicHermiteSpline::size() const {
    return segments_.size();
}

bool PiecewiseCubicHermiteSpline::empty() const {
    return segments_.empty();
}

}  // namespace autorigami
