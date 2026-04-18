#include "autorigami/validation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace autorigami {

namespace {

[[nodiscard]] double squared_distance(const Vec3& left, const Vec3& right) {
    return norm2(left - right);
}

[[nodiscard]] std::vector<double> compute_arclengths(const std::vector<Vec3>& points) {
    std::vector<double> arclengths(points.size(), 0.0);
    for (std::size_t index = 1; index < points.size(); ++index) {
        arclengths[index] = arclengths[index - 1] + norm(points[index] - points[index - 1]);
    }
    return arclengths;
}

[[nodiscard]] std::vector<double> compute_curvature(const std::vector<Vec3>& points) {
    std::vector<double> curvature(points.size(), 0.0);
    if (points.size() < 3) {
        return curvature;
    }

    constexpr double epsilon = 1e-12;
    for (std::size_t index = 1; index + 1 < points.size(); ++index) {
        const Vec3 ab = points[index] - points[index - 1];
        const Vec3 bc = points[index + 1] - points[index];
        const Vec3 ac = points[index + 1] - points[index - 1];

        const double a = norm(ab);
        const double b = norm(bc);
        const double c = norm(ac);
        const double denom = a * b * c;
        if (denom <= epsilon) {
            continue;
        }

        const double twice_area = norm(cross(ab, bc));
        const double local_curvature = 2.0 * twice_area / denom;
        if (local_curvature > epsilon) {
            curvature[index] = local_curvature;
        }
    }

    return curvature;
}

[[nodiscard]] ConstraintReport compute_separation_report(
    const std::vector<Vec3>& points,
    const std::vector<double>& arclengths,
    double separation_world,
    int neighbor_exclusion
) {
    int compliant_count = 0;
    const double separation_sq = separation_world * separation_world;

    for (std::size_t i = 0; i < points.size(); ++i) {
        double nearest_allowed_sq = std::numeric_limits<double>::infinity();
        bool found_candidate = false;

        for (std::size_t j = 0; j < points.size(); ++j) {
            if (i == j) {
                continue;
            }
            if (std::abs(static_cast<long long>(j) - static_cast<long long>(i)) <= neighbor_exclusion) {
                continue;
            }
            if (std::abs(arclengths[j] - arclengths[i]) < separation_world) {
                continue;
            }

            found_candidate = true;
            nearest_allowed_sq = std::min(nearest_allowed_sq, squared_distance(points[i], points[j]));
        }

        if (!found_candidate || nearest_allowed_sq >= separation_sq) {
            ++compliant_count;
        }
    }

    return ConstraintReport{
        .compliant_count = compliant_count,
        .total_count = static_cast<int>(points.size()),
    };
}

[[nodiscard]] ConstraintReport compute_curvature_report(
    const std::vector<double>& curvature,
    double max_curvature
) {
    int compliant_count = 0;
    for (double value : curvature) {
        if (value <= max_curvature) {
            ++compliant_count;
        }
    }

    return ConstraintReport{
        .compliant_count = compliant_count,
        .total_count = static_cast<int>(curvature.size()),
    };
}

}  // namespace

double ConstraintReport::ratio() const {
    if (total_count <= 0) {
        return 0.0;
    }
    return static_cast<double>(compliant_count) / static_cast<double>(total_count);
}

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

ValidationReport validate_polyline_constraints(
    const std::vector<Vec3>& points,
    double separation,
    double max_curvature,
    int neighbor_exclusion
) {
    if (points.size() < 3) {
        throw std::invalid_argument("need at least 3 points");
    }
    if (separation < 0.0) {
        throw std::invalid_argument("separation must be >= 0");
    }
    const std::vector<double> arclengths = compute_arclengths(points);
    const std::vector<double> curvature = compute_curvature(points);

    return ValidationReport{
        .separation =
            compute_separation_report(points, arclengths, separation, neighbor_exclusion),
        .curvature = compute_curvature_report(curvature, max_curvature),
    };
}

}  // namespace autorigami
