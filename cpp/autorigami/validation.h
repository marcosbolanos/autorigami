#pragma once

#include <array>
#include <vector>

namespace autorigami {

struct ConstraintReport {
    int compliant_count;
    int total_count;

    [[nodiscard]] double ratio() const;
};

struct ValidationReport {
    ConstraintReport separation;
    ConstraintReport curvature;
};

using Point3 = std::array<double, 3>;

[[nodiscard]] ValidationReport validate_polyline_constraints(
    const std::vector<Point3>& points,
    double world_to_nm,
    double separation_nm,
    double min_curvature_radius_nm,
    int neighbor_exclusion
);

}  // namespace autorigami
