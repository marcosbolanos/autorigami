#include "autorigami/geometry.h"

#include <algorithm>

#include <Eigen/Geometry>

namespace autorigami {
namespace {

constexpr float kEpsilon = 1e-12F;

[[nodiscard]] SegmentSegmentDistanceResult compute_segment_segment_distance(
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

    return { (closest_p - closest_q).norm(), closest_p, closest_q, s, t };
}

[[nodiscard]] Vector3 distance_increase_direction(
    const Polyline& polyline,
    const EdgeIndex first_index,
    const EdgeIndex second_index,
    const SegmentSegmentDistanceResult& distance
) {
    if (distance.distance > kEpsilon) {
        return (distance.closest_p - distance.closest_q) / distance.distance;
    }

    const Vector3 first_direction =
        polyline[first_index + 1] - polyline[first_index];
    const Vector3 second_direction =
        polyline[second_index + 1] - polyline[second_index];
    const Vector3 normal = first_direction.cross(second_direction);
    if (normal.norm() > kEpsilon) {
        return normal.normalized();
    }

    const Vector3 first_midpoint =
        0.5F * (polyline[first_index] + polyline[first_index + 1]);
    const Vector3 second_midpoint =
        0.5F * (polyline[second_index] + polyline[second_index + 1]);
    const Vector3 midpoint_delta = first_midpoint - second_midpoint;
    if (midpoint_delta.norm() > kEpsilon) {
        return midpoint_delta.normalized();
    }

    return Vector3::UnitX();
}

}  // namespace

std::vector<SegmentSegmentDistanceResult> segment_segment_distance(
    const Polyline& polyline,
    const std::vector<std::pair<EdgeIndex, EdgeIndex>>& candidate_pairs
) {
    std::vector<SegmentSegmentDistanceResult> distances;
    distances.reserve(candidate_pairs.size());

    for (const auto& [first_index, second_index] : candidate_pairs) {
        const Edge first = { polyline[first_index], polyline[first_index + 1] };
        const Edge second = { polyline[second_index], polyline[second_index + 1] };
        distances.push_back(compute_segment_segment_distance(first, second));
    }

    return distances;
}

SeparationCorrectionResult apply_separation_correction(
    Polyline polyline,
    const std::size_t passes,
    const std::vector<std::pair<EdgeIndex, EdgeIndex>>& candidate_pairs,
    const float min_distance,
    const float fixed_step,
    const std::array<bool, 3>& coordinate_mask,
    const bool reverse_order
) {
    std::size_t correction_count = 0;

    for (std::size_t pass = 0; pass < passes; ++pass) {
        std::size_t pass_correction_count = 0;
        const bool pass_reverse_order = reverse_order != (pass % 2 == 1);
        for (std::size_t offset = 0; offset < candidate_pairs.size(); ++offset) {
            const std::size_t pair_index = pass_reverse_order
                ? candidate_pairs.size() - 1 - offset
                : offset;
            const auto& [first_index, second_index] = candidate_pairs[pair_index];
            const Edge first = { polyline[first_index], polyline[first_index + 1] };
            const Edge second = { polyline[second_index], polyline[second_index + 1] };
            const SegmentSegmentDistanceResult distance =
                compute_segment_segment_distance(first, second);
            if (distance.distance >= min_distance) {
                continue;
            }

            Vector3 direction = distance_increase_direction(
                polyline,
                first_index,
                second_index,
                distance
            );
            for (Eigen::Index coordinate = 0;
                 coordinate < direction.size();
                 ++coordinate) {
                if (!coordinate_mask[static_cast<std::size_t>(coordinate)]) {
                    direction[coordinate] = 0.0F;
                }
            }

            const float first_weight = 1.0F - distance.first_parameter;
            const float second_weight = 1.0F - distance.second_parameter;
            const float squared_weight_sum =
                first_weight * first_weight
                + distance.first_parameter * distance.first_parameter
                + second_weight * second_weight
                + distance.second_parameter * distance.second_parameter;
            const float denominator = squared_weight_sum * direction.squaredNorm();
            if (denominator <= kEpsilon) {
                continue;
            }

            const Vector3 correction =
                ((min_distance - distance.distance + fixed_step) / denominator)
                * direction;
            polyline[first_index] += first_weight * correction;
            polyline[first_index + 1] += distance.first_parameter * correction;
            polyline[second_index] -= second_weight * correction;
            polyline[second_index + 1] -= distance.second_parameter * correction;
            ++pass_correction_count;
        }
        correction_count += pass_correction_count;
        if (pass_correction_count == 0) {
            break;
        }
    }

    return { std::move(polyline), correction_count };
}

}  // namespace autorigami
