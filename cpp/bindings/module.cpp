#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct ConstraintReport {
    int compliant_count;
    int total_count;

    [[nodiscard]] double ratio() const {
        if (total_count <= 0) {
            return 0.0;
        }
        return static_cast<double>(compliant_count) / static_cast<double>(total_count);
    }
};

struct ValidationReport {
    ConstraintReport separation;
    ConstraintReport curvature;
};

int add(int left, int right) {
    return left + right;
}

using Point = std::array<double, 3>;

[[nodiscard]] double squared_distance(const Point& left, const Point& right) {
    const double dx = left[0] - right[0];
    const double dy = left[1] - right[1];
    const double dz = left[2] - right[2];
    return dx * dx + dy * dy + dz * dz;
}

[[nodiscard]] Point subtract(const Point& left, const Point& right) {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

[[nodiscard]] Point cross(const Point& left, const Point& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

[[nodiscard]] double norm(const Point& point) {
    return std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
}

[[nodiscard]] std::vector<Point> load_points(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& points
) {
    if (points.ndim() != 2 || points.shape(1) != 3) {
        throw std::invalid_argument("points must have shape (N, 3)");
    }
    if (points.shape(0) < 3) {
        throw std::invalid_argument("need at least 3 points");
    }

    const auto view = points.unchecked<2>();
    std::vector<Point> result(static_cast<std::size_t>(points.shape(0)));
    for (py::ssize_t row = 0; row < points.shape(0); ++row) {
        result[static_cast<std::size_t>(row)] = {view(row, 0), view(row, 1), view(row, 2)};
    }
    return result;
}

[[nodiscard]] std::vector<double> compute_arclengths(const std::vector<Point>& points) {
    std::vector<double> arclengths(points.size(), 0.0);
    for (std::size_t index = 1; index < points.size(); ++index) {
        arclengths[index] = arclengths[index - 1] + norm(subtract(points[index], points[index - 1]));
    }
    return arclengths;
}

[[nodiscard]] std::vector<double> compute_curvature_radius(const std::vector<Point>& points) {
    std::vector<double> radius(points.size(), std::numeric_limits<double>::infinity());
    if (points.size() < 3) {
        return radius;
    }

    constexpr double epsilon = 1e-12;
    for (std::size_t index = 1; index + 1 < points.size(); ++index) {
        const Point ab = subtract(points[index], points[index - 1]);
        const Point bc = subtract(points[index + 1], points[index]);
        const Point ac = subtract(points[index + 1], points[index - 1]);

        const double a = norm(ab);
        const double b = norm(bc);
        const double c = norm(ac);
        const double denom = a * b * c;
        if (denom <= epsilon) {
            continue;
        }

        const double twice_area = norm(cross(ab, bc));
        const double curvature = 2.0 * twice_area / denom;
        if (curvature > epsilon) {
            radius[index] = 1.0 / curvature;
        }
    }

    return radius;
}

[[nodiscard]] ConstraintReport compute_separation_report(
    const std::vector<Point>& points,
    const std::vector<double>& arclengths,
    double separation_world,
    int neighbor_exclusion
) {
    int compliant_count = 0;
    const double separation_sq = separation_world * separation_world;

    for (std::size_t i = 0; i < points.size(); ++i) {
        double nearest_allowed_sq = std::numeric_limits<double>::infinity();
        bool found_candidate = false;

        for (std::size_t j = 0; j < points.size(); ++j) {
            if (i == j) {
                continue;
            }
            if (std::abs(static_cast<long long>(j) - static_cast<long long>(i)) <= neighbor_exclusion) {
                continue;
            }
            if (std::abs(arclengths[j] - arclengths[i]) < separation_world) {
                continue;
            }

            found_candidate = true;
            nearest_allowed_sq = std::min(nearest_allowed_sq, squared_distance(points[i], points[j]));
        }

        if (!found_candidate || nearest_allowed_sq >= separation_sq) {
            ++compliant_count;
        }
    }

    return ConstraintReport{
        .compliant_count = compliant_count,
        .total_count = static_cast<int>(points.size()),
    };
}

[[nodiscard]] ConstraintReport compute_curvature_report(
    const std::vector<double>& radius,
    double min_radius_world
) {
    int compliant_count = 0;
    for (double value : radius) {
        if (value >= min_radius_world) {
            ++compliant_count;
        }
    }

    return ConstraintReport{
        .compliant_count = compliant_count,
        .total_count = static_cast<int>(radius.size()),
    };
}

[[nodiscard]] ValidationReport validate_polyline_constraints(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& points,
    double world_to_nm,
    double separation_nm,
    double min_curvature_radius_nm,
    int neighbor_exclusion
) {
    if (world_to_nm <= 0.0) {
        throw std::invalid_argument("world_to_nm must be > 0");
    }

    const std::vector<Point> point_data = load_points(points);
    const double separation_world = separation_nm / world_to_nm;
    const double min_radius_world = min_curvature_radius_nm / world_to_nm;
    const std::vector<double> arclengths = compute_arclengths(point_data);
    const std::vector<double> radius = compute_curvature_radius(point_data);

    return ValidationReport{
        .separation =
            compute_separation_report(point_data, arclengths, separation_world, neighbor_exclusion),
        .curvature = compute_curvature_report(radius, min_radius_world),
    };
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native C++ extensions for autorigami.";

    py::class_<ConstraintReport>(module, "ConstraintReport")
        .def_readonly("compliant_count", &ConstraintReport::compliant_count)
        .def_readonly("total_count", &ConstraintReport::total_count)
        .def_property_readonly("ratio", &ConstraintReport::ratio);

    py::class_<ValidationReport>(module, "ValidationReport")
        .def_readonly("separation", &ValidationReport::separation)
        .def_readonly("curvature", &ValidationReport::curvature);

    module.def("add", &add, py::arg("left"), py::arg("right"));
    module.def(
        "validate_polyline_constraints",
        &validate_polyline_constraints,
        py::arg("points"),
        py::arg("world_to_nm"),
        py::arg("separation_nm"),
        py::arg("min_curvature_radius_nm"),
        py::arg("neighbor_exclusion") = 8
    );
}
