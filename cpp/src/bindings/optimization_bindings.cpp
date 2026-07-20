#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "autorigami/optimization.h"

namespace py = pybind11;

namespace {

[[nodiscard]] py::dict evaluation_dict(
    const autorigami::TangentPointEvaluation& evaluation
) {
    py::dict output;
    output["energy"] = evaluation.energy;
    output["repulsive_energy"] = evaluation.repulsive_energy;
    output["attractive_energy"] = evaluation.attractive_energy;
    output["differential"] = evaluation.differential;
    output["exact_pair_count"] = evaluation.exact_pair_count;
    output["approximated_cluster_count"] =
        evaluation.approximated_cluster_count;
    return output;
}

[[nodiscard]] autorigami::TangentPointParameters parameters(
    const double target_distance,
    const double attraction_strength,
    const double local_exclusion_length
) {
    return { target_distance, attraction_strength, local_exclusion_length };
}

}  // namespace

void register_optimization_bindings(py::module_& module) {
    module.def(
        "evaluate_tangent_point_exact",
        [](const autorigami::Polyline& polyline,
           const double target_distance,
           const double attraction_strength,
           const double local_exclusion_length) {
            return evaluation_dict(autorigami::evaluate_tangent_point_exact(
                polyline,
                parameters(
                    target_distance,
                    attraction_strength,
                    local_exclusion_length
                )
            ));
        },
        py::arg("polyline"),
        py::arg("target_distance"),
        py::arg("attraction_strength"),
        py::arg("local_exclusion_length")
    );
    module.def(
        "evaluate_tangent_point_hierarchical",
        [](const autorigami::Polyline& polyline,
           const double target_distance,
           const double attraction_strength,
           const double local_exclusion_length,
           const double opening_angle,
           const std::size_t leaf_size) {
            return evaluation_dict(
                autorigami::evaluate_tangent_point_hierarchical(
                    polyline,
                    parameters(
                        target_distance,
                        attraction_strength,
                        local_exclusion_length
                    ),
                    opening_angle,
                    leaf_size
                )
            );
        },
        py::arg("polyline"),
        py::arg("target_distance"),
        py::arg("attraction_strength"),
        py::arg("local_exclusion_length"),
        py::arg("opening_angle") = 0.25,
        py::arg("leaf_size") = 8
    );
}
