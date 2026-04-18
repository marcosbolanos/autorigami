#include "autorigami/curvature_math.h"

#include "autorigami/interval.h"
#include "autorigami/ivec3.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace autorigami {

namespace {

struct TimeIntervalPowers {
    Interval t;
    Interval t_squared;
};

struct CurvatureSquaredBounds {
    double lower;
    double upper;
};

class CurvatureMaximizer {
  public:
    CurvatureMaximizer(
        const CubicPowerBasisSegment& segment,
        const CurvatureMaxSolverConfig& config
    )
        : segment_(segment), config_(config) {}

    [[nodiscard]] CurvatureMaxResult solve() {
        const double curvature_squared_at_start = segment_.curvature_squared(0.0);
        const double curvature_squared_at_end = segment_.curvature_squared(1.0);

        best_curvature_squared_ = std::max(curvature_squared_at_start, curvature_squared_at_end);
        best_t_ = curvature_squared_at_start >= curvature_squared_at_end ? 0.0 : 1.0;

        const double upper_curvature_squared = recurse(0.0, 1.0, 0);

        return {
            .lower_bound = std::sqrt(std::max(0.0, best_curvature_squared_)),
            .upper_bound = std::sqrt(std::max(0.0, upper_curvature_squared)),
            .best_t = best_t_,
        };
    }

  private:
    [[nodiscard]] static TimeIntervalPowers make_time_interval(double lower, double upper) {
        const Interval t(lower, upper);
        return {
            .t = t,
            .t_squared = square(t),
        };
    }

    [[nodiscard]] IVec3 first_derivative_interval(double lower, double upper) const {
        const TimeIntervalPowers time = make_time_interval(lower, upper);
        return as_interval(segment_.a) * (Interval(3.0) * time.t_squared) +
               as_interval(segment_.b) * (Interval(2.0) * time.t) +
               as_interval(segment_.c);
    }

    [[nodiscard]] IVec3 second_derivative_interval(double lower, double upper) const {
        const TimeIntervalPowers time = make_time_interval(lower, upper);
        return as_interval(segment_.a) * (Interval(6.0) * time.t) +
               as_interval(segment_.b) * Interval(2.0);
    }

    [[nodiscard]] Interval curvature_squared_interval(double lower, double upper) const {
        const IVec3 velocity = first_derivative_interval(lower, upper);
        const IVec3 acceleration = second_derivative_interval(lower, upper);

        const Interval numerator = norm2(cross(velocity, acceleration));
        const Interval velocity_norm_squared = norm2(velocity);
        if (velocity_norm_squared.lo <= 0.0) {
            return {0.0, std::numeric_limits<double>::infinity()};
        }

        const Interval denominator =
            velocity_norm_squared * velocity_norm_squared * velocity_norm_squared;
        return numerator / denominator;
    }

    [[nodiscard]] double recurse(double lower, double upper, int depth) {
        const double midpoint = 0.5 * (lower + upper);

        const double curvature_squared_at_lower = segment_.curvature_squared(lower);
        const double curvature_squared_at_upper = segment_.curvature_squared(upper);
        const double curvature_squared_at_midpoint = segment_.curvature_squared(midpoint);

        update_best(curvature_squared_at_lower, lower);
        update_best(curvature_squared_at_upper, upper);
        update_best(curvature_squared_at_midpoint, midpoint);

        const Interval curvature_interval = curvature_squared_interval(lower, upper);
        const double local_lower = std::max({
            curvature_squared_at_lower,
            curvature_squared_at_upper,
            curvature_squared_at_midpoint,
        });
        const double local_upper = curvature_interval.hi;

        if (local_upper <= best_curvature_squared_ + config_.abs_tolerance_squared) {
            return local_upper;
        }

        if (depth >= config_.max_depth ||
            (upper - lower) <= config_.min_interval_width ||
            (local_upper - local_lower) <= config_.abs_tolerance_squared) {
            return local_upper;
        }

        const double left_upper = recurse(lower, midpoint, depth + 1);
        const double right_upper = recurse(midpoint, upper, depth + 1);
        return std::max(left_upper, right_upper);
    }

    void update_best(double curvature_squared_value, double t) {
        if (curvature_squared_value > best_curvature_squared_) {
            best_curvature_squared_ = curvature_squared_value;
            best_t_ = t;
        }
    }

    const CubicPowerBasisSegment& segment_;
    const CurvatureMaxSolverConfig& config_;
    double best_curvature_squared_ = 0.0;
    double best_t_ = 0.0;
};

}  // namespace

CurvatureMaxResult max_curvature_of_segment(
    const CubicPowerBasisSegment& segment,
    const CurvatureMaxSolverConfig& config
) {
    return CurvatureMaximizer(segment, config).solve();
}

}  // namespace autorigami
