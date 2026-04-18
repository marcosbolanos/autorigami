#pragma once

#include <algorithm>
#include <limits>

namespace autorigami {

struct Interval {
    double lo;
    double hi;

    constexpr Interval() noexcept
        : lo(0.0), hi(0.0) {}

    constexpr Interval(double value) noexcept
        : lo(value), hi(value) {}

    constexpr Interval(double lower, double upper) noexcept
        : lo(std::min(lower, upper)), hi(std::max(lower, upper)) {}
};

[[nodiscard]] inline constexpr Interval hull(const Interval& left, const Interval& right) noexcept {
    return {std::min(left.lo, right.lo), std::max(left.hi, right.hi)};
}

[[nodiscard]] inline constexpr Interval operator+(const Interval& left, const Interval& right) noexcept {
    return {left.lo + right.lo, left.hi + right.hi};
}

[[nodiscard]] inline constexpr Interval operator-(const Interval& left, const Interval& right) noexcept {
    return {left.lo - right.hi, left.hi - right.lo};
}

[[nodiscard]] inline constexpr Interval operator-(const Interval& value) noexcept {
    return {-value.hi, -value.lo};
}

[[nodiscard]] inline constexpr Interval operator*(const Interval& left, const Interval& right) noexcept {
    const double p1 = left.lo * right.lo;
    const double p2 = left.lo * right.hi;
    const double p3 = left.hi * right.lo;
    const double p4 = left.hi * right.hi;
    return {
        std::min({p1, p2, p3, p4}),
        std::max({p1, p2, p3, p4}),
    };
}

[[nodiscard]] inline constexpr Interval operator*(const Interval& value, double scale) noexcept {
    return value * Interval(scale);
}

[[nodiscard]] inline constexpr Interval operator*(double scale, const Interval& value) noexcept {
    return Interval(scale) * value;
}

[[nodiscard]] inline Interval operator/(const Interval& numerator, const Interval& denominator) noexcept {
    if (denominator.lo <= 0.0 && denominator.hi >= 0.0) {
        const double infinity = std::numeric_limits<double>::infinity();
        return {-infinity, infinity};
    }
    return numerator * Interval(1.0 / denominator.hi, 1.0 / denominator.lo);
}

[[nodiscard]] inline constexpr Interval square(const Interval& value) noexcept {
    if (value.lo >= 0.0) {
        return {value.lo * value.lo, value.hi * value.hi};
    }
    if (value.hi <= 0.0) {
        return {value.hi * value.hi, value.lo * value.lo};
    }
    const double max_magnitude_squared = std::max(value.lo * value.lo, value.hi * value.hi);
    return {0.0, max_magnitude_squared};
}

[[nodiscard]] inline constexpr Interval cube(const Interval& value) noexcept {
    return {value.lo * value.lo * value.lo, value.hi * value.hi * value.hi};
}

}  // namespace autorigami
