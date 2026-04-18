#pragma once

#include "autorigami/interval.h"
#include "autorigami/vec3.h"

namespace autorigami {

struct IVec3 {
    Interval x;
    Interval y;
    Interval z;
};

[[nodiscard]] inline constexpr IVec3 operator+(const IVec3& left, const IVec3& right) noexcept {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

[[nodiscard]] inline constexpr IVec3 operator-(const IVec3& left, const IVec3& right) noexcept {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

[[nodiscard]] inline constexpr IVec3 operator*(const IVec3& value, const Interval& scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

[[nodiscard]] inline constexpr IVec3 operator*(const Interval& scale, const IVec3& value) noexcept {
    return value * scale;
}

[[nodiscard]] inline constexpr Interval dot(const IVec3& left, const IVec3& right) noexcept {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] inline constexpr IVec3 cross(const IVec3& left, const IVec3& right) noexcept {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

[[nodiscard]] inline constexpr Interval norm2(const IVec3& value) noexcept {
    return square(value.x) + square(value.y) + square(value.z);
}

[[nodiscard]] inline constexpr IVec3 as_interval(const Vec3& value) noexcept {
    return {Interval(value.x), Interval(value.y), Interval(value.z)};
}

}  // namespace autorigami
