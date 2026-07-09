#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_geometry_bindings(py::module_& module);

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native C++ extensions for autorigami.";
    register_geometry_bindings(module);
}
