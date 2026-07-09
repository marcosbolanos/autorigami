#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "autorigami/geometry.h"

namespace py = pybind11;

void register_geometry_bindings(py::module_& module) {
    module.def(
        "segment_segment_distance",
        &autorigami::segment_segment_distance,
        py::arg("polyline"),
        py::arg("candidate_pairs")
    );
}
