#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autorigami/validation/nonlocal_distance.h"
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

autorigami::NonlocalDistanceValidationResult validate_polyline_nonlocal_distance_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& points,
    double minimum_separation,
    double nonlocal_window,
    bool stop_on_first_violation
) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (N, 3)");
    }
    if (points.shape(0) < 2) {
        throw std::invalid_argument("points must have at least 2 rows");
    }

    std::vector<autorigami::Vec3> loaded_points;
    loaded_points.reserve(static_cast<std::size_t>(points.shape(0)));
    const auto view = points.unchecked<2>();
    for (py::ssize_t row = 0; row < points.shape(0); ++row) {
        loaded_points.push_back(load_vec3_row(view, row));
    }

    return autorigami::validate_polyline_nonlocal_distance(
        loaded_points,
        minimum_separation,
        nonlocal_window,
        stop_on_first_violation
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
    module.def(
        "validate_polyline_nonlocal_distance",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& points,
           double minimum_separation,
           double nonlocal_window,
           bool stop_on_first_violation) {
            const autorigami::NonlocalDistanceValidationResult result =
                validate_polyline_nonlocal_distance_py(
                    points,
                    minimum_separation,
                    nonlocal_window,
                    stop_on_first_violation
                );
            py::dict out;
            out["violation_count"] = result.violation_count;
            out["minimum_checked_distance"] = result.minimum_checked_distance;
            return out;
        },
        py::arg("points"),
        py::arg("minimum_separation"),
        py::arg("nonlocal_window"),
        py::arg("stop_on_first_violation") = false
    );
}
