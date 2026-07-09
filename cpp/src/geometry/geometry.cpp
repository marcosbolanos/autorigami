#include "autorigami/geometry.h"

#include <algorithm>

namespace autorigami {
namespace {

constexpr float kEpsilon = 1e-12F;

[[nodiscard]] SegmentSegmentDistance compute_segment_segment_distance(
    const Edge& first,
    const Edge& second
) {
    const Vector3& p0 = first[0];
    const Vector3& p1 = first[1];
    const Vector3& q0 = second[0];
    const Vector3& q1 = second[1];

    const Vector3 u = p1 - p0;
    const Vector3 v = q1 - q0;
    const Vector3 w = p0 - q0;

    const float a = u.dot(u);
    const float b = u.dot(v);
    const float c = v.dot(v);
    const float d = u.dot(w);
    const float e = v.dot(w);
    const float denom = a * c - b * b;

    float s = 0.0F;
    float t = 0.0F;

    if (denom > kEpsilon) {
        s = (b * e - c * d) / denom;
        t = (a * e - b * d) / denom;
    } else {
        s = 0.0F;
        t = c > kEpsilon ? e / c : 0.0F;
    }

    s = std::clamp(s, 0.0F, 1.0F);
    t = std::clamp(t, 0.0F, 1.0F);

    if (c > kEpsilon) {
        t = std::clamp(v.dot(p0 + s * u - q0) / c, 0.0F, 1.0F);
    }
    if (a > kEpsilon) {
        s = std::clamp(u.dot(q0 + t * v - p0) / a, 0.0F, 1.0F);
    }

    const Vector3 closest_p = p0 + s * u;
    const Vector3 closest_q = q0 + t * v;

    return { (closest_p - closest_q).norm(), closest_p, closest_q };
}

}  // namespace

std::vector<SegmentSegmentDistance> segment_segment_distance(
    const std::vector<std::pair<Edge, Edge>>& candidate_pairs
) {
    std::vector<SegmentSegmentDistance> distances;
    distances.reserve(candidate_pairs.size());

    for (const auto& [first, second] : candidate_pairs) {
        distances.push_back(compute_segment_segment_distance(first, second));
    }

    return distances;
}

}  // namespace autorigami
