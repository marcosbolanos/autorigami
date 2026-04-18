#pragma once

#include <vector>

#include "autorigami/vec3.h"

namespace autorigami {

enum class TangentPolicy {
    CatmullRomCentripetal,
};

struct CubicPowerBasisSegment {
    Vec3 a;
    Vec3 b;
    Vec3 c;
    Vec3 d;

    [[nodiscard]] bool operator==(const CubicPowerBasisSegment& other) const = default;
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

    [[nodiscard]] const std::vector<CubicHermiteSegment>& segments() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] bool empty() const;

  private:
    std::vector<CubicHermiteSegment> segments_;
};

}  // namespace autorigami
