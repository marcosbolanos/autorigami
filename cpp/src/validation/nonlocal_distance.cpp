#include "autorigami/validation/nonlocal_distance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace autorigami {

namespace {

constexpr double kSmallEpsilon = 1e-12;

[[nodiscard]] Vec3 subtract(const Vec3& a, const Vec3& b) {
    return {.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
}

[[nodiscard]] Vec3 add(const Vec3& a, const Vec3& b) {
    return {.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z};
}

[[nodiscard]] Vec3 scale(const Vec3& value, double factor) {
    return {.x = value.x * factor, .y = value.y * factor, .z = value.z * factor};
}

[[nodiscard]] double segment_segment_distance_squared(
    const Vec3& p0,
    const Vec3& p1,
    const Vec3& q0,
    const Vec3& q1
) {
    const Vec3 u = subtract(p1, p0);
    const Vec3 v = subtract(q1, q0);
    const Vec3 w = subtract(p0, q0);

    const double a = dot(u, u);
    const double b = dot(u, v);
    const double c = dot(v, v);
    const double d = dot(u, w);
    const double e = dot(v, w);

    const double denom = a * c - b * b;
    double s_num = 0.0;
    double s_den = denom;
    double t_num = 0.0;
    double t_den = denom;

    if (denom < kSmallEpsilon) {
        s_num = 0.0;
        s_den = 1.0;
        t_num = e;
        t_den = c;
    } else {
        s_num = b * e - c * d;
        t_num = a * e - b * d;

        if (s_num < 0.0) {
            s_num = 0.0;
            t_num = e;
            t_den = c;
        } else if (s_num > s_den) {
            s_num = s_den;
            t_num = e + b;
            t_den = c;
        }
    }

    if (t_num < 0.0) {
        t_num = 0.0;
        if (-d < 0.0) {
            s_num = 0.0;
        } else if (-d > a) {
            s_num = s_den;
        } else {
            s_num = -d;
            s_den = a;
        }
    } else if (t_num > t_den) {
        t_num = t_den;
        if ((-d + b) < 0.0) {
            s_num = 0.0;
        } else if ((-d + b) > a) {
            s_num = s_den;
        } else {
            s_num = -d + b;
            s_den = a;
        }
    }

    const double s = (std::abs(s_num) < kSmallEpsilon) ? 0.0 : s_num / s_den;
    const double t = (std::abs(t_num) < kSmallEpsilon) ? 0.0 : t_num / t_den;

    const Vec3 delta = subtract(add(w, scale(u, s)), scale(v, t));
    return dot(delta, delta);
}

[[nodiscard]] double segment_arc_gap(double a0, double a1, double b0, double b1) {
    if (a1 <= b0) {
        return b0 - a1;
    }
    if (b1 <= a0) {
        return a0 - b1;
    }
    return 0.0;
}

}  // namespace

NonlocalDistanceValidationResult validate_polyline_nonlocal_distance(
    const std::vector<Vec3>& points,
    double minimum_separation,
    double nonlocal_window,
    bool stop_on_first_violation
) {
    if (points.size() < 2) {
        throw std::invalid_argument("points must have at least 2 entries");
    }
    if (minimum_separation <= 0.0) {
        throw std::invalid_argument("minimum_separation must be > 0");
    }
    if (nonlocal_window <= 0.0) {
        throw std::invalid_argument("nonlocal_window must be > 0");
    }

    std::vector<double> cumulative(points.size(), 0.0);
    for (std::size_t i = 1; i < points.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + norm(subtract(points[i], points[i - 1]));
    }

    int violation_count = 0;
    double minimum_checked_distance = std::numeric_limits<double>::infinity();
    const double minimum_separation_sq = minimum_separation * minimum_separation;

    for (std::size_t i = 0; i + 1 < points.size(); ++i) {
        const Vec3& a0 = points[i];
        const Vec3& a1 = points[i + 1];
        const double arc_a0 = cumulative[i];
        const double arc_a1 = cumulative[i + 1];

        for (std::size_t j = 0; j < i; ++j) {
            const double arc_b0 = cumulative[j];
            const double arc_b1 = cumulative[j + 1];
            if (segment_arc_gap(arc_a0, arc_a1, arc_b0, arc_b1) < nonlocal_window) {
                continue;
            }

            const Vec3& b0 = points[j];
            const Vec3& b1 = points[j + 1];
            const double distance_sq = segment_segment_distance_squared(a0, a1, b0, b1);
            minimum_checked_distance = std::min(minimum_checked_distance, std::sqrt(std::max(distance_sq, 0.0)));
            if (distance_sq < minimum_separation_sq) {
                violation_count += 1;
                if (stop_on_first_violation) {
                    return NonlocalDistanceValidationResult{
                        .violation_count = violation_count,
                        .minimum_checked_distance = minimum_checked_distance,
                    };
                }
            }
        }
    }

    return NonlocalDistanceValidationResult{
        .violation_count = violation_count,
        .minimum_checked_distance = minimum_checked_distance,
    };
}

}  // namespace autorigami
