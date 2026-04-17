#include "autorigami/validation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace autorigami {

namespace {

[[nodiscard]] double squared_distance(const Point3& left, const Point3& right) {
    const double dx = left[0] - right[0];
    const double dy = left[1] - right[1];
    const double dz = left[2] - right[2];
    return dx * dx + dy * dy + dz * dz;
}

[[nodiscard]] Point3 subtract(const Point3& left, const Point3& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

[[nodiscard]] Point3 cross(const Point3& left, const Point3& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

[[nodiscard]] double norm(const Point3& point) {
    return std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
}

[[nodiscard]] std::vector<double> compute_arclengths(const std::vector<Point3>& points) {
    std::vector<double> arclengths(points.size(), 0.0);
    for (std::size_t index = 1; index < points.size(); ++index) {
        arclengths[index] = arclengths[index - 1] + norm(subtract(points[index], points[index - 1]));
    }
    return arclengths;
}

[[nodiscard]] std::vector<double> compute_curvature_radius(const std::vector<Point3>& points) {
    std::vector<double> radius(points.size(), std::numeric_limits<double>::infinity());
    if (points.size() < 3) {
        return radius;
    }

    constexpr double epsilon = 1e-12;
    for (std::size_t index = 1; index + 1 < points.size(); ++index) {
        const Point3 ab = subtract(points[index], points[index - 1]);
        const Point3 bc = subtract(points[index + 1], points[index]);
        const Point3 ac = subtract(points[index + 1], points[index - 1]);

        const double a = norm(ab);
        const double b = norm(bc);
        const double c = norm(ac);
        const double denom = a * b * c;
        if (denom <= epsilon) {
            continue;
        }

        const double twice_area = norm(cross(ab, bc));
        const double curvature = 2.0 * twice_area / denom;
        if (curvature > epsilon) {
            radius[index] = 1.0 / curvature;
        }
    }

    return radius;
}

[[nodiscard]] ConstraintReport compute_separation_report(
    const std::vector<Point3>& points,
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
    const std::vector<double>& radius,
    double min_radius_world
) {
    int compliant_count = 0;
    for (double value : radius) {
        if (value >= min_radius_world) {
            ++compliant_count;
        }
    }

    return ConstraintReport{
        .compliant_count = compliant_count,
        .total_count = static_cast<int>(radius.size()),
    };
}

}  // namespace

double ConstraintReport::ratio() const {
    if (total_count <= 0) {
        return 0.0;
    }
    return static_cast<double>(compliant_count) / static_cast<double>(total_count);
}

ValidationReport validate_polyline_constraints(
    const std::vector<Point3>& points,
    double world_to_nm,
    double separation_nm,
    double min_curvature_radius_nm,
    int neighbor_exclusion
) {
    if (points.size() < 3) {
        throw std::invalid_argument("need at least 3 points");
    }
    if (world_to_nm <= 0.0) {
        throw std::invalid_argument("world_to_nm must be > 0");
    }

    const double separation_world = separation_nm / world_to_nm;
    const double min_radius_world = min_curvature_radius_nm / world_to_nm;
    const std::vector<double> arclengths = compute_arclengths(points);
    const std::vector<double> radius = compute_curvature_radius(points);

    return ValidationReport{
        .separation =
            compute_separation_report(points, arclengths, separation_world, neighbor_exclusion),
        .curvature = compute_curvature_report(radius, min_radius_world),
    };
}

}  // namespace autorigami
