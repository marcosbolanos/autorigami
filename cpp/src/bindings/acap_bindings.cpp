#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "autorigami/acap/toolpath_stats.h"
#include "autorigami/vec3.h"

namespace py = pybind11;

namespace {

template <typename View>
[[nodiscard]] autorigami::Vec3 load_vec3_row(const View& view, py::ssize_t row) {
    return {.x = view(row, 0), .y = view(row, 1), .z = view(row, 2)};
}

[[nodiscard]] std::vector<autorigami::Vec3> load_polyline_points(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& points
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
    return loaded_points;
}

}  // namespace

void register_acap_bindings(py::module_& module) {
    module.def(
        "compute_acap_toolpath_stats",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& points,
           double minimum_separation_world,
           double nonlocal_window_world) {
            const autorigami::acap::ToolpathStats stats = autorigami::acap::compute_toolpath_stats(
                load_polyline_points(points),
                minimum_separation_world,
                nonlocal_window_world
            );
            py::dict out;
            out["point_count"] = stats.point_count;
            out["length_world"] = stats.length_world;
            out["nonlocal_violation_count"] = stats.nonlocal_distance.violation_count;
            out["minimum_checked_distance_world"] = stats.nonlocal_distance.minimum_checked_distance;
            return out;
        },
        py::arg("points"),
        py::arg("minimum_separation_world"),
        py::arg("nonlocal_window_world")
    );
}
