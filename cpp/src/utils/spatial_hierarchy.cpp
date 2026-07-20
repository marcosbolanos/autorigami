#include "autorigami/utils.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace autorigami::utils {

bool SpatialTreeNode::is_leaf() const { return left == nullptr; }

double SpatialTreeNode::spatial_radius() const {
    return 0.5 * (maximum_position - minimum_position).norm();
}

double SpatialTreeNode::tangent_radius() const {
    return 0.5 * (maximum_tangent - minimum_tangent).norm();
}

SpatialHierarchy::SpatialHierarchy(
    std::vector<SpatialSample> samples,
    const std::size_t leaf_size,
    const HierarchySplit split
) : samples_(std::move(samples)), leaf_size_(leaf_size), split_(split) {
    if (samples_.empty() || leaf_size_ == 0) {
        throw std::invalid_argument("spatial hierarchy inputs must be nonempty");
    }
    permutation_.resize(samples_.size());
    std::iota(permutation_.begin(), permutation_.end(), 0);
    root_ = build(0, samples_.size());
}

const std::vector<SpatialSample>& SpatialHierarchy::samples() const {
    return samples_;
}

std::span<const std::size_t> SpatialHierarchy::indices(
    const SpatialTreeNode& node
) const {
    return std::span(permutation_).subspan(node.begin, node.end - node.begin);
}

const SpatialTreeNode& SpatialHierarchy::root() const { return *root_; }

std::unique_ptr<SpatialTreeNode> SpatialHierarchy::build(
    const std::size_t begin,
    const std::size_t end
) {
    auto node = std::make_unique<SpatialTreeNode>();
    node->begin = begin;
    node->end = end;
    node->minimum_position.setConstant(std::numeric_limits<double>::infinity());
    node->maximum_position.setConstant(-std::numeric_limits<double>::infinity());
    node->minimum_tangent.setConstant(std::numeric_limits<double>::infinity());
    node->maximum_tangent.setConstant(-std::numeric_limits<double>::infinity());
    node->minimum_coordinate = std::numeric_limits<double>::infinity();
    node->maximum_coordinate = -std::numeric_limits<double>::infinity();
    for (std::size_t offset = begin; offset < end; ++offset) {
        const SpatialSample& sample = samples_[permutation_[offset]];
        node->minimum_position = node->minimum_position.cwiseMin(sample.position);
        node->maximum_position = node->maximum_position.cwiseMax(sample.position);
        node->minimum_tangent = node->minimum_tangent.cwiseMin(sample.tangent);
        node->maximum_tangent = node->maximum_tangent.cwiseMax(sample.tangent);
        node->center += sample.mass * sample.position;
        node->average_tangent += sample.mass * sample.tangent;
        node->total_mass += sample.mass;
        node->minimum_coordinate = std::min(
            node->minimum_coordinate, sample.coordinate
        );
        node->maximum_coordinate = std::max(
            node->maximum_coordinate, sample.coordinate
        );
    }
    node->center /= node->total_mass;
    node->average_tangent /= node->total_mass;
    if (node->average_tangent.norm() > 0.0) {
        node->average_tangent.normalize();
    }
    if (end - begin <= leaf_size_) {
        return node;
    }

    const std::size_t middle = begin + (end - begin) / 2;
    if (split_ == HierarchySplit::Coordinate) {
        std::nth_element(
            permutation_.begin() + static_cast<std::ptrdiff_t>(begin),
            permutation_.begin() + static_cast<std::ptrdiff_t>(middle),
            permutation_.begin() + static_cast<std::ptrdiff_t>(end),
            [&](const std::size_t first, const std::size_t second) {
                return samples_[first].coordinate < samples_[second].coordinate;
            }
        );
    } else {
        Eigen::Matrix<double, 6, 1> spans;
        spans.head<3>() = node->maximum_position - node->minimum_position;
        spans.tail<3>() = node->maximum_tangent - node->minimum_tangent;
        Eigen::Index axis = 0;
        spans.maxCoeff(&axis);
        std::nth_element(
            permutation_.begin() + static_cast<std::ptrdiff_t>(begin),
            permutation_.begin() + static_cast<std::ptrdiff_t>(middle),
            permutation_.begin() + static_cast<std::ptrdiff_t>(end),
            [&](const std::size_t first, const std::size_t second) {
                const SpatialSample& a = samples_[first];
                const SpatialSample& b = samples_[second];
                return axis < 3 ? a.position[axis] < b.position[axis]
                                : a.tangent[axis - 3] < b.tangent[axis - 3];
            }
        );
    }
    node->left = build(begin, middle);
    node->right = build(middle, end);
    return node;
}

bool clusters_are_admissible(
    const SpatialTreeNode& first,
    const SpatialTreeNode& second,
    const double opening_angle
) {
    const double distance = (first.center - second.center).norm();
    return distance > 0.0
        && (first.spatial_radius() + second.spatial_radius()) / distance
            <= opening_angle
        && std::max(first.tangent_radius(), second.tangent_radius())
            <= opening_angle;
}

}  // namespace autorigami::utils
