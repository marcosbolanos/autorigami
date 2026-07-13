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
            const std::vector<std::pair<autorigami::EdgeIndex, autorigami::EdgeIndex>>& candidate_pairs
        ) {
            const auto distances = autorigami::segment_segment_distance(polyline, candidate_pairs);

            py::list output;
            for (const auto& distance : distances) {
                output.append(py::make_tuple(
                    distance.distance,
                    distance.closest_p,
                    distance.closest_q
                ));
            }
            return output;
        },
        py::arg("polyline"),
        py::arg("candidate_pairs")
    );
    module.def(
        "apply_separation_correction",
        [](const autorigami::Polyline& polyline,
           const std::size_t passes,
           const std::vector<std::pair<autorigami::EdgeIndex, autorigami::EdgeIndex>>& candidate_pairs,
           const float min_distance,
           const float fixed_step,
           const std::array<bool, 3>& coordinate_mask,
           const bool reverse_order) {
            const auto result = autorigami::apply_separation_correction(
                polyline,
                passes,
                candidate_pairs,
                min_distance,
                fixed_step,
                coordinate_mask,
                reverse_order
            );
            return py::make_tuple(result.polyline, result.correction_count);
        },
        py::arg("polyline"),
        py::arg("passes"),
        py::arg("candidate_pairs"),
        py::arg("min_distance"),
        py::arg("fixed_step") = 0.0F,
        py::arg("coordinate_mask") = std::array<bool, 3>{ true, true, true },
        py::arg("reverse_order") = false
    );
}
