#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_generator_bindings(py::module_& module);
void register_validation_bindings(py::module_& module);

namespace {

int add(int left, int right) {
    return left + right;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native C++ extensions for autorigami.";
    module.def("add", &add, py::arg("left"), py::arg("right"));
    register_generator_bindings(module);
    register_validation_bindings(module);
}
