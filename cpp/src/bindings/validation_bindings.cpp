#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autorigami/validation.h"

namespace py = pybind11;

namespace {

template <typename View>
[[nodiscard]] autorigami::Vec3 load_vec3_row(const View& view, py::ssize_t row) {
    return {.x = view(row, 0), .y = view(row, 1), .z = view(row, 2)};
}

[[nodiscard]] autorigami::PiecewiseCubicHermiteSpline load_piecewise_hermite(
    const py::object& piecewise_hermite
) {
    const py::array_t<double, py::array::c_style | py::array::forcecast> points =
        piecewise_hermite.attr("points")
            .cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    const py::array_t<double, py::array::c_style | py::array::forcecast> tangents =
        piecewise_hermite.attr("tangents")
            .cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();

    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("piecewise_hermite.points must have shape (N, 3)");
    }
    if (tangents.ndim() != 2 || tangents.shape(1) != 3) {
        throw std::invalid_argument("piecewise_hermite.tangents must have shape (N, 3)");
    }
    if (points.shape(0) != tangents.shape(0)) {
        throw std::invalid_argument("piecewise_hermite points/tangents row counts must match");
    }
    if (points.shape(0) < 2) {
        throw std::invalid_argument("piecewise_hermite must have at least two points");
    }

    const auto points_view = points.unchecked<2>();
    const auto tangents_view = tangents.unchecked<2>();

    std::vector<autorigami::CubicHermiteSegment> loaded_segments;
    loaded_segments.reserve(static_cast<std::size_t>(points.shape(0) - 1));
    for (py::ssize_t index = 0; index + 1 < points.shape(0); ++index) {
        loaded_segments.push_back({
            .p0 = load_vec3_row(points_view, index),
            .p1 = load_vec3_row(points_view, index + 1),
            .m0 = load_vec3_row(tangents_view, index),
            .m1 = load_vec3_row(tangents_view, index + 1),
        });
    }

    return autorigami::PiecewiseCubicHermiteSpline(std::move(loaded_segments));
}

bool validate_piecewise_curve_curvature_py(
    const py::object& piecewise_hermite,
    double max_curvature,
    double curvature_tolerance
) {
    return autorigami::validate_piecewise_curve_curvature(
        load_piecewise_hermite(piecewise_hermite),
        max_curvature,
        curvature_tolerance
    );
}

}  // namespace

void register_validation_bindings(py::module_& module) {
    module.def(
        "validate_piecewise_curve_curvature",
        &validate_piecewise_curve_curvature_py,
        py::arg("piecewise_hermite"),
        py::arg("max_curvature"),
        py::arg("curvature_tolerance")
    );
}
