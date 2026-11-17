#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "autorigami/simulation.h"

namespace py = pybind11;

void register_simulation_bindings(py::module_& module) {
    py::class_<autorigami::AxialGravitySimulation>(
        module, "AxialGravitySimulation"
    )
        .def(
            py::init([](const autorigami::PointMatrix& points,
                        const autorigami::PinnedMask& pinned,
                        const autorigami::CurveIndices& curve_indices,
                        const float collision_radius,
                        const std::size_t ignored_adjacent_vertices,
                        const float tension_stiffness,
                        const float tension_damping,
                        const float floor_height,
                        const float gravity,
                        const float linear_damping,
                        const float maximum_depenetration_velocity,
                        const float maximum_linear_velocity,
                        const unsigned int position_iterations,
                        const unsigned int velocity_iterations,
                        const unsigned int thread_count) {
                return std::make_unique<autorigami::AxialGravitySimulation>(
                    points,
                    pinned,
                    curve_indices,
                    autorigami::AxialGravityParameters{
                        collision_radius,
                        ignored_adjacent_vertices,
                        tension_stiffness,
                        tension_damping,
                        floor_height,
                        gravity,
                        linear_damping,
                        maximum_depenetration_velocity,
                        maximum_linear_velocity,
                        position_iterations,
                        velocity_iterations,
                        thread_count,
                    }
                );
            }),
            py::kw_only(),
            py::arg("points"),
            py::arg("pinned"),
            py::arg("curve_indices"),
            py::arg("collision_radius"),
            py::arg("ignored_adjacent_vertices"),
            py::arg("tension_stiffness"),
            py::arg("tension_damping"),
            py::arg("floor_height"),
            py::arg("gravity"),
            py::arg("linear_damping"),
            py::arg("maximum_depenetration_velocity"),
            py::arg("maximum_linear_velocity"),
            py::arg("position_iterations"),
            py::arg("velocity_iterations"),
            py::arg("thread_count")
        )
        .def(
            "step",
            [](autorigami::AxialGravitySimulation& simulation,
               const std::size_t step_count,
               const float time_step) {
                const py::gil_scoped_release release;
                simulation.step(step_count, time_step);
            },
            py::arg("step_count"),
            py::arg("time_step")
        )
        .def(
            "set_gravity",
            &autorigami::AxialGravitySimulation::set_gravity,
            py::arg("gravity")
        )
        .def("points", &autorigami::AxialGravitySimulation::points)
        .def(
            "maximum_speed",
            &autorigami::AxialGravitySimulation::maximum_speed
        );
}
