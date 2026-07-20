#include "autorigami/optimization.h"
#include "autorigami/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace autorigami {
namespace {

using Vector3d = Eigen::Vector3d;

struct EdgeData {
    Vector3d vector;
    Vector3d midpoint;
    Vector3d tangent;
    double length;
    double midpoint_arclength;
};

struct RatioDifferential {
    double ratio;
    Vector3d point;
    Vector3d other;
    Vector3d tangent;
};

struct DirectedAccumulation {
    double energy = 0.0;
    double repulsive_energy = 0.0;
    Eigen::MatrixXd differential;
    std::size_t exact_pair_count = 0;
    std::size_t approximated_cluster_count = 0;

    explicit DirectedAccumulation(const std::size_t vertex_count)
        : differential(Eigen::MatrixXd::Zero(
              static_cast<Eigen::Index>(vertex_count),
              3
          )) {}
};

[[nodiscard]] std::vector<Vector3d> double_points(const Polyline& polyline) {
    std::vector<Vector3d> points;
    points.reserve(polyline.size());
    for (const Vector3& point : polyline) {
        points.push_back(point.cast<double>());
    }
    return points;
}

[[nodiscard]] std::vector<EdgeData> edge_data(
    const std::vector<Vector3d>& points
) {
    std::vector<EdgeData> edges;
    edges.reserve(points.size() - 1);
    double arclength = 0.0;
    for (std::size_t index = 0; index + 1 < points.size(); ++index) {
        const Vector3d vector = points[index + 1] - points[index];
        const double length = vector.norm();
        if (!(length > 0.0)) {
            throw std::invalid_argument("polyline edges must have positive length");
        }
        edges.push_back({
            vector,
            0.5 * (points[index] + points[index + 1]),
            vector / length,
            length,
            arclength + 0.5 * length,
        });
        arclength += length;
    }
    return edges;
}

[[nodiscard]] RatioDifferential ratio_differential(
    const Vector3d& point,
    const Vector3d& other,
    const Vector3d& tangent,
    const double target_radius
) {
    const Vector3d displacement = point - other;
    const double squared_distance = displacement.squaredNorm();
    if (!(squared_distance > 0.0)) {
        throw std::invalid_argument("tangent-point samples must be distinct");
    }
    const double tangent_projection = tangent.dot(displacement);
    const Vector3d normal = displacement - tangent_projection * tangent;
    const double squared_normal = std::max(0.0, normal.squaredNorm());
    const double scale = 4.0 * target_radius * target_radius;
    const double ratio = scale * squared_normal
        / (squared_distance * squared_distance);
    const Vector3d displacement_derivative = scale
        * (2.0 * normal / std::pow(squared_distance, 2)
           - 4.0 * squared_normal * displacement
               / std::pow(squared_distance, 3));
    const Vector3d tangent_derivative = -2.0 * scale * tangent_projection
        * displacement / std::pow(squared_distance, 2);
    return {
        ratio,
        displacement_derivative,
        -displacement_derivative,
        tangent_derivative,
    };
}

[[nodiscard]] double kernel(const double ratio, const bool attractive) {
    return ratio * ratio - 2.0 * static_cast<double>(attractive) * ratio;
}

void accumulate_source_sample(
    const Vector3d& source,
    const Vector3d& target,
    const Vector3d& source_tangent,
    const Vector3d& target_tangent,
    const double target_radius,
    const bool attractive,
    const double scale,
    double& pair_energy,
    double& pair_repulsive,
    Vector3d& source_derivative,
    Vector3d& source_tangent_derivative
) {
    const RatioDifferential forward = ratio_differential(
        source, target, source_tangent, target_radius
    );
    const RatioDifferential reverse = ratio_differential(
        target, source, target_tangent, target_radius
    );
    pair_energy += scale
        * (kernel(forward.ratio, attractive)
           + kernel(reverse.ratio, attractive));
    pair_repulsive += scale
        * (kernel(forward.ratio, false) + kernel(reverse.ratio, false));
    const double forward_factor = 2.0 * forward.ratio
        - 2.0 * static_cast<double>(attractive);
    const double reverse_factor = 2.0 * reverse.ratio
        - 2.0 * static_cast<double>(attractive);
    source_derivative += scale
        * (forward_factor * forward.point + reverse_factor * reverse.other);
    source_tangent_derivative += scale * forward_factor * forward.tangent;
}

void accumulate_source_edge(
    const std::vector<Vector3d>& points,
    const std::vector<EdgeData>& edges,
    const std::size_t source_index,
    const std::span<const Vector3d> target_samples,
    const Vector3d& target_tangent,
    const double target_mass,
    const TangentPointParameters& parameters,
    const bool attractive,
    DirectedAccumulation& result
) {
    const EdgeData& source_edge = edges[source_index];
    const double target_radius = 0.5 * parameters.target_distance;
    double pair_energy = 0.0;
    double pair_repulsive = 0.0;
    std::array<Vector3d, 2> endpoint_derivatives = {
        Vector3d::Zero(), Vector3d::Zero()
    };
    Vector3d tangent_derivative = Vector3d::Zero();
    const double sample_scale = parameters.attraction_strength
        * source_edge.length * target_mass
        / (2.0 * static_cast<double>(target_samples.size()));

    for (std::size_t source_local = 0; source_local < 2; ++source_local) {
        for (const Vector3d& target : target_samples) {
            accumulate_source_sample(
                points[source_index + source_local],
                target,
                source_edge.tangent,
                target_tangent,
                target_radius,
                attractive,
                sample_scale,
                pair_energy,
                pair_repulsive,
                endpoint_derivatives[source_local],
                tangent_derivative
            );
        }
    }

    const Eigen::Matrix3d tangent_projector = (
        Eigen::Matrix3d::Identity()
        - source_edge.tangent * source_edge.tangent.transpose()
    ) / source_edge.length;
    const Vector3d edge_derivative = tangent_projector * tangent_derivative
        + (pair_energy / source_edge.length) * source_edge.tangent;
    endpoint_derivatives[0] -= edge_derivative;
    endpoint_derivatives[1] += edge_derivative;
    for (std::size_t local = 0; local < 2; ++local) {
        result.differential.row(static_cast<Eigen::Index>(source_index + local))
            += endpoint_derivatives[local].transpose();
    }
    result.energy += pair_energy;
    result.repulsive_energy += pair_repulsive;
}

[[nodiscard]] bool all_attractive(
    const utils::SpatialTreeNode& node,
    const double source_arclength,
    const double exclusion
) {
    return node.maximum_coordinate < source_arclength - exclusion
        || node.minimum_coordinate > source_arclength + exclusion;
}

[[nodiscard]] bool admissible(
    const utils::SpatialTreeNode& node,
    const EdgeData& source,
    const double opening_angle
) {
    const double distance = (source.midpoint - node.center).norm();
    if (!(distance > 0.0)) {
        return false;
    }
    return node.spatial_radius() / distance <= opening_angle
        && node.tangent_radius() <= opening_angle;
}

void traverse_source(
    const std::vector<Vector3d>& points,
    const std::vector<EdgeData>& edges,
    const utils::SpatialHierarchy& hierarchy,
    const utils::SpatialTreeNode& node,
    const std::size_t source_index,
    const TangentPointParameters& parameters,
    const double opening_angle,
    DirectedAccumulation& result
) {
    const EdgeData& source = edges[source_index];
    const bool attractive = all_attractive(
        node, source.midpoint_arclength, parameters.local_exclusion_length
    );
    if (!node.is_leaf() && attractive
        && admissible(node, source, opening_angle)) {
        const std::array<Vector3d, 1> target = { node.center };
        accumulate_source_edge(
            points,
            edges,
            source_index,
            target,
            node.average_tangent,
            node.total_mass,
            parameters,
            true,
            result
        );
        ++result.approximated_cluster_count;
        return;
    }
    if (!node.is_leaf()) {
        traverse_source(
            points, edges, hierarchy, *node.left, source_index,
            parameters, opening_angle, result
        );
        traverse_source(
            points, edges, hierarchy, *node.right, source_index,
            parameters, opening_angle, result
        );
        return;
    }
    for (const std::size_t target_index : hierarchy.indices(node)) {
        if (source_index == target_index
            || std::max(source_index, target_index)
                    - std::min(source_index, target_index)
                <= 1) {
            continue;
        }
        const std::array<Vector3d, 2> target = {
            points[target_index], points[target_index + 1]
        };
        const bool pair_attractive = std::abs(
            source.midpoint_arclength
            - edges[target_index].midpoint_arclength
        ) > parameters.local_exclusion_length;
        accumulate_source_edge(
            points,
            edges,
            source_index,
            target,
            edges[target_index].tangent,
            edges[target_index].length,
            parameters,
            pair_attractive,
            result
        );
        ++result.exact_pair_count;
    }
}

[[nodiscard]] TangentPointEvaluation finish(DirectedAccumulation result) {
    return {
        0.5 * result.energy,
        0.5 * result.repulsive_energy,
        0.5 * (result.energy - result.repulsive_energy),
        std::move(result.differential),
        result.exact_pair_count / 2,
        result.approximated_cluster_count,
    };
}

}  // namespace

TangentPointEvaluation evaluate_tangent_point_exact(
    const Polyline& polyline,
    const TangentPointParameters& parameters
) {
    if (polyline.size() < 4) {
        throw std::invalid_argument("polyline must contain at least four vertices");
    }
    const std::vector<Vector3d> points = double_points(polyline);
    const std::vector<EdgeData> edges = edge_data(points);
    DirectedAccumulation result(polyline.size());
    for (std::size_t source = 0; source < edges.size(); ++source) {
        for (std::size_t target = 0; target < edges.size(); ++target) {
            if (source == target
                || std::max(source, target) - std::min(source, target) <= 1) {
                continue;
            }
            const std::array<Vector3d, 2> samples = {
                points[target], points[target + 1]
            };
            const bool attractive = std::abs(
                edges[source].midpoint_arclength
                - edges[target].midpoint_arclength
            ) > parameters.local_exclusion_length;
            accumulate_source_edge(
                points, edges, source, samples, edges[target].tangent,
                edges[target].length, parameters, attractive, result
            );
            ++result.exact_pair_count;
        }
    }
    return finish(std::move(result));
}

TangentPointEvaluation evaluate_tangent_point_hierarchical(
    const Polyline& polyline,
    const TangentPointParameters& parameters,
    const double opening_angle,
    const std::size_t leaf_size
) {
    if (!(opening_angle > 0.0) || leaf_size == 0) {
        throw std::invalid_argument(
            "opening_angle and leaf_size must be positive"
        );
    }
    const std::vector<Vector3d> points = double_points(polyline);
    const std::vector<EdgeData> edges = edge_data(points);
    std::vector<utils::SpatialSample> samples;
    samples.reserve(edges.size());
    for (const EdgeData& edge : edges) {
        samples.push_back({
            edge.midpoint,
            edge.tangent,
            edge.length,
            edge.midpoint_arclength,
        });
    }
    const utils::SpatialHierarchy hierarchy(std::move(samples), leaf_size);
    DirectedAccumulation result(polyline.size());
    for (std::size_t source = 0; source < edges.size(); ++source) {
        traverse_source(
            points, edges, hierarchy, hierarchy.root(), source, parameters,
            opening_angle, result
        );
    }
    return finish(std::move(result));
}

}  // namespace autorigami
