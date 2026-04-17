#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "autorigami/validation.h"

namespace py = pybind11;

namespace {

[[nodiscard]] std::vector<autorigami::Point3> load_points(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& points
) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (N, 3)");
    }

    const auto view = points.unchecked<2>();
    std::vector<autorigami::Point3> result(static_cast<std::size_t>(points.shape(0)));
    for (py::ssize_t row = 0; row < points.shape(0); ++row) {
        result[static_cast<std::size_t>(row)] = {view(row, 0), view(row, 1), view(row, 2)};
    }
    return result;
}

autorigami::ValidationReport validate_polyline_constraints_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& points,
    double world_to_nm,
    double separation_nm,
    double min_curvature_radius_nm,
    int neighbor_exclusion
) {
    return autorigami::validate_polyline_constraints(
        load_points(points),
        world_to_nm,
        separation_nm,
        min_curvature_radius_nm,
        neighbor_exclusion
    );
}

}  // namespace

void register_validation_bindings(py::module_& module) {
    py::class_<autorigami::ConstraintReport>(module, "ConstraintReport")
        .def_readonly("compliant_count", &autorigami::ConstraintReport::compliant_count)
        .def_readonly("total_count", &autorigami::ConstraintReport::total_count)
        .def_property_readonly("ratio", &autorigami::ConstraintReport::ratio);

    py::class_<autorigami::ValidationReport>(module, "ValidationReport")
        .def_readonly("separation", &autorigami::ValidationReport::separation)
        .def_readonly("curvature", &autorigami::ValidationReport::curvature);

    module.def(
        "validate_polyline_constraints",
        &validate_polyline_constraints_py,
        py::arg("points"),
        py::arg("world_to_nm"),
        py::arg("separation_nm"),
        py::arg("min_curvature_radius_nm"),
        py::arg("neighbor_exclusion") = 8
    );
}
