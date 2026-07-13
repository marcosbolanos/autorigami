#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "autorigami/geometry.h"

namespace py = pybind11;

void register_geometry_bindings(py::module_& module) {
    module.def(
        "segment_segment_distance",
        [](
            const autorigami::Polyline& polyline,
            const std::vector<std::pair<autorigami::EdgeIndex, autorigami::EdgeIndex>>& candidate_pairs,
            const bool include_optimization_data
        ) {
            const auto distances = autorigami::segment_segment_distance(polyline, candidate_pairs);

            py::list output;
            for (const auto& distance : distances) {
                if (include_optimization_data) {
                    output.append(py::make_tuple(
                        distance.distance,
                        distance.closest_p,
                        distance.closest_q,
                        distance.first_parameter,
                        distance.second_parameter
                    ));
                } else {
                    output.append(py::make_tuple(
                        distance.distance,
                        distance.closest_p,
                        distance.closest_q
                    ));
                }
            }
            return output;
        },
        py::arg("polyline"),
        py::arg("candidate_pairs"),
        py::arg("include_optimization_data") = false
    );
}
