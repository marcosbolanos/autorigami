#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace autorigami {

using Vector3 = Eigen::Vector3f;
using Edge = std::array<Vector3, 2>;
using Polyline = std::vector<Vector3>;
using EdgeIndex = std::size_t;
struct SegmentSegmentDistanceResult {
    float distance;
    Vector3 closest_p;
    Vector3 closest_q;
    float first_parameter;
    float second_parameter;
};
struct SeparationCorrectionResult {
    Polyline polyline;
    std::size_t correction_count;
};

[[nodiscard]] std::vector<SegmentSegmentDistanceResult> segment_segment_distance(
    const Polyline& polyline,
    const std::vector<std::pair<EdgeIndex, EdgeIndex>>& candidate_pairs
);

[[nodiscard]] SeparationCorrectionResult apply_separation_correction(
    Polyline polyline,
    std::size_t passes,
    const std::vector<std::pair<EdgeIndex, EdgeIndex>>& candidate_pairs,
    float min_distance,
    float fixed_step,
    const std::array<bool, 3>& coordinate_mask,
    bool reverse_order
);

}  // namespace autorigami
