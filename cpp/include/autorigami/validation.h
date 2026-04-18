#pragma once

#include <vector>

#include "autorigami/vec3.h"

namespace autorigami {

struct ConstraintReport {
    int compliant_count;
    int total_count;

    [[nodiscard]] double ratio() const;
    [[nodiscard]] bool operator==(const ConstraintReport& other) const = default;
};

struct ValidationReport {
    ConstraintReport separation;
    ConstraintReport curvature;

    [[nodiscard]] bool operator==(const ValidationReport& other) const = default;
};

[[nodiscard]] ValidationReport validate_polyline_constraints(
    const std::vector<Vec3>& points,
    double world_to_nm,
    double separation_nm,
    double min_curvature_radius_nm,
    int neighbor_exclusion
);

}  // namespace autorigami
