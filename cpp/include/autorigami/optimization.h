#pragma once

#include <cstddef>
#include <Eigen/Core>

#include "autorigami/geometry.h"

namespace autorigami {

struct TangentPointParameters {
    double target_distance;
    double attraction_strength;
    double local_exclusion_length;
};

struct TangentPointEvaluation {
    double energy;
    double repulsive_energy;
    double attractive_energy;
    Eigen::MatrixXd differential;
    std::size_t exact_pair_count;
    std::size_t approximated_cluster_count;
};

[[nodiscard]] TangentPointEvaluation evaluate_tangent_point_exact(
    const Polyline& polyline,
    const TangentPointParameters& parameters
);

[[nodiscard]] TangentPointEvaluation evaluate_tangent_point_hierarchical(
    const Polyline& polyline,
    const TangentPointParameters& parameters,
    double opening_angle,
    std::size_t leaf_size
);

}  // namespace autorigami
