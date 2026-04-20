#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>

#include "autorigami/generator.h"

namespace py = pybind11;

namespace {

[[nodiscard]] py::array_t<double> vec3_vector_to_numpy(const std::vector<autorigami::Vec3>& values) {
    py::array_t<double> output({static_cast<py::ssize_t>(values.size()), py::ssize_t{3}});
    auto mutable_view = output.mutable_unchecked<2>();
    for (std::size_t row = 0; row < values.size(); ++row) {
        const autorigami::Vec3& value = values[row];
        mutable_view(static_cast<py::ssize_t>(row), 0) = value.x;
        mutable_view(static_cast<py::ssize_t>(row), 1) = value.y;
        mutable_view(static_cast<py::ssize_t>(row), 2) = value.z;
    }
    return output;
}

[[nodiscard]] py::tuple piecewise_hermite_generator_py() {
    const autorigami::PiecewiseHermiteGeneratorResult generated = autorigami::piecewise_hermite_generator();

    py::module_ parametrization_module = py::module_::import("autorigami.parametrization");
    py::object piecewise_hermite_class = parametrization_module.attr("PiecewiseHermite");

    py::dict kwargs;
    kwargs["points"] = vec3_vector_to_numpy(generated.piecewise_hermite.points);
    kwargs["tangents"] = vec3_vector_to_numpy(generated.piecewise_hermite.tangents);

    py::dict run_data;
    run_data["cpp_point_count"] = generated.run_data.point_count;
    run_data["cpp_segment_count"] = generated.run_data.segment_count;
    run_data["cpp_parameter_step"] = generated.run_data.parameter_step;

    return py::make_tuple(piecewise_hermite_class(**kwargs), run_data);
}

}  // namespace

void register_generator_bindings(py::module_& module) {
    module.def("piecewise_hermite_generator", &piecewise_hermite_generator_py);
}
