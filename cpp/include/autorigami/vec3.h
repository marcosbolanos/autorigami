#pragma once

#include <cmath>
#include <stdexcept>

namespace autorigami {

struct Vec3 {
    double x;
    double y;
    double z;

    [[nodiscard]] constexpr Vec3 operator+(const Vec3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    [[nodiscard]] constexpr Vec3 operator-(const Vec3& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    [[nodiscard]] constexpr Vec3 operator*(double scale) const noexcept {
        return {x * scale, y * scale, z * scale};
    }

    [[nodiscard]] constexpr Vec3 operator/(double scale) const {
        if (scale == 0.0) {
            throw std::invalid_argument("Vec3 division by zero");
        }
        return {x / scale, y / scale, z / scale};
    }

    [[nodiscard]] bool operator==(const Vec3& other) const = default;
};

[[nodiscard]] inline constexpr Vec3 operator*(double scale, const Vec3& value) noexcept {
    return value * scale;
}

[[nodiscard]] inline constexpr double dot(const Vec3& left, const Vec3& right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] inline constexpr Vec3 cross(const Vec3& left, const Vec3& right) noexcept {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

[[nodiscard]] inline constexpr double norm2(const Vec3& value) noexcept {
    return dot(value, value);
}

[[nodiscard]] inline double norm(const Vec3& value) {
    return std::sqrt(norm2(value));
}

}  // namespace autorigami
