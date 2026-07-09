#pragma once

#include <array>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace autorigami {

using Vector3 = Eigen::Vector3f;
using Edge = std::array<Vector3, 2>;
using SegmentSegmentDistance = std::tuple<float, Vector3, Vector3>;

[[nodiscard]] std::vector<SegmentSegmentDistance> segment_segment_distance(
    const std::vector<std::pair<Edge, Edge>>& candidate_pairs
);

}  // namespace autorigami
