#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

int add(int left, int right) {
    return left + right;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native C++ extensions for autorigami.";

    module.def("add", &add, py::arg("left"), py::arg("right"));
}
