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
using SegmentSegmentDistance = std::tuple<float, Vector3, Vector3>;

[[nodiscard]] std::vector<SegmentSegmentDistance> segment_segment_distance(
    const Polyline& polyline,
    const std::vector<std::pair<EdgeIndex, EdgeIndex>>& candidate_pairs
);

}  // namespace autorigami
