#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "autorigami/generator.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

namespace py = pybind11;

namespace {

using geometrycentral::Vector3;
using geometrycentral::surface::ManifoldSurfaceMesh;
using geometrycentral::surface::VertexPositionGeometry;

struct ConvertedManifoldMesh {
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
};

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

[[nodiscard]] std::vector<Vector3> load_vertex_positions(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& vertices
) {
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
        throw std::invalid_argument("vertices must have shape (V, 3)");
    }

    std::vector<Vector3> output;
    output.reserve(static_cast<std::size_t>(vertices.shape(0)));

    const auto view = vertices.unchecked<2>();
    for (py::ssize_t index = 0; index < vertices.shape(0); ++index) {
        output.emplace_back(view(index, 0), view(index, 1), view(index, 2));
    }
    return output;
}

[[nodiscard]] autorigami::Vec3 load_axis_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& axis
) {
    if (axis.ndim() != 1 || axis.shape(0) != 3) {
        throw std::invalid_argument("axis must have shape (3,)");
    }

    const auto view = axis.unchecked<1>();
    return {.x = view(0), .y = view(1), .z = view(2)};
}

[[nodiscard]] std::vector<std::vector<std::size_t>> load_triangle_faces(
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& faces,
    std::size_t vertex_count
) {
    if (faces.ndim() != 2 || faces.shape(1) != 3) {
        throw std::invalid_argument("faces must have shape (F, 3)");
    }

    std::vector<std::vector<std::size_t>> output;
    output.reserve(static_cast<std::size_t>(faces.shape(0)));

    const auto view = faces.unchecked<2>();
    for (py::ssize_t face_index = 0; face_index < faces.shape(0); ++face_index) {
        std::vector<std::size_t> triangle;
        triangle.reserve(3);
        for (py::ssize_t corner = 0; corner < 3; ++corner) {
            const std::int64_t raw_index = view(face_index, corner);
            if (raw_index < 0) {
                throw std::invalid_argument("faces must contain non-negative vertex indices");
            }
            const std::size_t vertex_index = static_cast<std::size_t>(raw_index);
            if (vertex_index >= vertex_count) {
                throw std::invalid_argument("faces contain vertex indices out of bounds");
            }
            triangle.push_back(vertex_index);
        }
        output.push_back(std::move(triangle));
    }
    return output;
}

[[nodiscard]] ConvertedManifoldMesh convert_trimesh_to_manifold_surface_mesh(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& vertices,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& faces
) {
    std::vector<Vector3> vertex_positions = load_vertex_positions(vertices);
    std::vector<std::vector<std::size_t>> polygons =
        load_triangle_faces(faces, static_cast<std::size_t>(vertex_positions.size()));

    auto [mesh, geometry] =
        geometrycentral::surface::makeManifoldSurfaceMeshAndGeometry(polygons, vertex_positions);

    return ConvertedManifoldMesh{
        .mesh = std::move(mesh),
        .geometry = std::move(geometry),
    };
}

[[nodiscard]] py::dict convert_trimesh_to_manifold_surface_mesh_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& vertices,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& faces
) {
    ConvertedManifoldMesh converted = convert_trimesh_to_manifold_surface_mesh(vertices, faces);

    py::dict out;
    out["vertex_count"] = converted.mesh->nVertices();
    out["face_count"] = converted.mesh->nFaces();
    return out;
}

[[nodiscard]] py::tuple piecewise_hermite_generator_py(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& vertices,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& faces,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& axis
) {
    ConvertedManifoldMesh converted = convert_trimesh_to_manifold_surface_mesh(vertices, faces);
    const autorigami::Vec3 axis_vector = load_axis_vector(axis);
    const autorigami::PiecewiseHermiteGeneratorResult generated = autorigami::piecewise_hermite_generator(
        *converted.mesh,
        *converted.geometry,
        axis_vector
    );

    py::module_ parametrization_module = py::module_::import("autorigami.parametrization");
    py::object piecewise_hermite_class = parametrization_module.attr("PiecewiseHermite");

    py::dict kwargs;
    kwargs["points"] = vec3_vector_to_numpy(generated.piecewise_hermite.points);
    kwargs["tangents"] = vec3_vector_to_numpy(generated.piecewise_hermite.tangents);

    py::dict run_data;
    run_data["cpp_point_count"] = generated.run_data.point_count;
    run_data["cpp_segment_count"] = generated.run_data.segment_count;
    run_data["cpp_parameter_step"] = generated.run_data.parameter_step;
    run_data["input_mesh_vertex_count"] = converted.mesh->nVertices();
    run_data["input_mesh_face_count"] = converted.mesh->nFaces();
    run_data["input_axis_x"] = axis_vector.x;
    run_data["input_axis_y"] = axis_vector.y;
    run_data["input_axis_z"] = axis_vector.z;

    return py::make_tuple(piecewise_hermite_class(**kwargs), run_data);
}

}  // namespace

void register_generator_bindings(py::module_& module) {
    module.def(
        "convert_trimesh_to_manifold_surface_mesh",
        &convert_trimesh_to_manifold_surface_mesh_py,
        py::arg("vertices"),
        py::arg("faces")
    );
    module.def(
        "piecewise_hermite_generator",
        &piecewise_hermite_generator_py,
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("axis")
    );
}
