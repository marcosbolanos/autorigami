#pragma once

#include <vector>

#include "autorigami/vec3.h"

namespace autorigami {

enum class TangentPolicy {
    CatmullRomCentripetal,
};

struct CubicPowerBasisSegment {
    Vec3 a; // r(t) = a t³ + b t² + c t + d
    Vec3 b;
    Vec3 c;
    Vec3 d;

    [[nodiscard]] bool operator==(const CubicPowerBasisSegment& other) const = default;

    [[nodiscard]] Vec3 position(double t) const {
        return ((a * t + b) * t + c) * t + d;
    }

    [[nodiscard]] Vec3 first_derivative(double t) const {
        return (3.0 * a * t + 2.0 * b) * t + c;
    }

    [[nodiscard]] Vec3 second_derivative(double t) const {
        return 6.0 * a * t + 2.0 * b;
    }

    [[nodiscard]] Vec3 third_derivative() const {
        return 6.0 * a;
    }

    // squared curvature of a segment k²(t), in bounds [0, 1]
    [[nodiscard]] double curvature_squared(double t) const;

    // from paper : reduced form  of H(t)
    // reduced polynomial whose roots identify stationary curvature points
    [[nodiscard]] double reduced_curvature_extremum_polynomial(double t) const;
};

struct CubicHermiteSegment {
    Vec3 p0;
    Vec3 p1;
    Vec3 m0;
    Vec3 m1;

    [[nodiscard]] CubicPowerBasisSegment to_power_basis() const;
    [[nodiscard]] bool operator==(const CubicHermiteSegment& other) const = default;
};

class PiecewiseCubicHermiteSpline {
  public:
    explicit PiecewiseCubicHermiteSpline(std::vector<CubicHermiteSegment> segments);

    [[nodiscard]] static PiecewiseCubicHermiteSpline from_polyline(
        const std::vector<Vec3>& points,
        TangentPolicy tangent_policy
    );

    [[nodiscard]] const std::vector<CubicHermiteSegment>& segments() const {
        return segments_;
    }

    [[nodiscard]] std::size_t size() const {
        return segments_.size();
    }

    [[nodiscard]] bool empty() const {
        return segments_.empty();
    }

  private:
    std::vector<CubicHermiteSegment> segments_;
};
}  // namespace autorigami
