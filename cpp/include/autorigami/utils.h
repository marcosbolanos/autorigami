#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace autorigami::utils {

enum class HierarchySplit { SpatialTangent, Coordinate };

struct SpatialSample {
    Eigen::Vector3d position;
    Eigen::Vector3d tangent;
    double mass;
    double coordinate;
};

struct SpatialTreeNode {
    std::size_t begin = 0;
    std::size_t end = 0;
    Eigen::Vector3d minimum_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d maximum_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d minimum_tangent = Eigen::Vector3d::Zero();
    Eigen::Vector3d maximum_tangent = Eigen::Vector3d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d average_tangent = Eigen::Vector3d::Zero();
    double total_mass = 0.0;
    double minimum_coordinate = 0.0;
    double maximum_coordinate = 0.0;
    std::unique_ptr<SpatialTreeNode> left;
    std::unique_ptr<SpatialTreeNode> right;

    [[nodiscard]] bool is_leaf() const;
    [[nodiscard]] double spatial_radius() const;
    [[nodiscard]] double tangent_radius() const;
};

class SpatialHierarchy {
  public:
    SpatialHierarchy(
        std::vector<SpatialSample> samples,
        std::size_t leaf_size,
        HierarchySplit split = HierarchySplit::SpatialTangent
    );

    [[nodiscard]] const std::vector<SpatialSample>& samples() const;
    [[nodiscard]] std::span<const std::size_t> indices(
        const SpatialTreeNode& node
    ) const;
    [[nodiscard]] const SpatialTreeNode& root() const;

  private:
    [[nodiscard]] std::unique_ptr<SpatialTreeNode> build(
        std::size_t begin,
        std::size_t end
    );

    std::vector<SpatialSample> samples_;
    std::vector<std::size_t> permutation_;
    std::size_t leaf_size_;
    HierarchySplit split_;
    std::unique_ptr<SpatialTreeNode> root_;
};

[[nodiscard]] bool clusters_are_admissible(
    const SpatialTreeNode& first,
    const SpatialTreeNode& second,
    double opening_angle
);

}  // namespace autorigami::utils
