#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "autorigami/validation.h"

namespace py = pybind11;

namespace {

[[nodiscard]] autorigami::PiecewiseCubicHermiteSpline load_hermite_segments(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& segments
) {
    if (segments.ndim() != 3 || segments.shape(1) != 4 || segments.shape(2) != 3) {
        throw std::invalid_argument("segments must have shape (N, 4, 3)");
    }

    const auto view = segments.unchecked<3>();
    std::vector<autorigami::CubicHermiteSegment> loaded_segments(
        static_cast<std::size_t>(segments.shape(0))
    );
    for (py::ssize_t index = 0; index < segments.shape(0); ++index) {
        loaded_segments[static_cast<std::size_t>(index)] = {
            .p0 = {.x = view(index, 0, 0), .y = view(index, 0, 1), .z = view(index, 0, 2)},
            .p1 = {.x = view(index, 1, 0), .y = view(index, 1, 1), .z = view(index, 1, 2)},
            .m0 = {.x = view(index, 2, 0), .y = view(index, 2, 1), .z = view(index, 2, 2)},
            .m1 = {.x = view(index, 3, 0), .y = view(index, 3, 1), .z = view(index, 3, 2)},
        };
    }
    return autorigami::PiecewiseCubicHermiteSpline(std::move(loaded_segments));
}

bool validate_piecewise_curve_curvature_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& segments,
    double max_curvature,
    double curvature_tolerance
) {
    return autorigami::validate_piecewise_curve_curvature(
        load_hermite_segments(segments),
        max_curvature,
        curvature_tolerance
    );
}

}  // namespace

void register_validation_bindings(py::module_& module) {
    module.def(
        "validate_piecewise_curve_curvature",
        &validate_piecewise_curve_curvature_py,
        py::arg("segments"),
        py::arg("max_curvature"),
        py::arg("curvature_tolerance")
    );
}
