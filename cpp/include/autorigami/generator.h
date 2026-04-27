#pragma once

#include <cstddef>
#include <vector>

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/vector3.h"

namespace autorigami {

struct PiecewiseHermiteData {
    std::vector<geometrycentral::Vector3> points;
    std::vector<geometrycentral::Vector3> tangents;
};

struct GeneratorAxis {
    geometrycentral::Vector3 origin;
    geometrycentral::Vector3 direction;
};

struct PiecewiseHermiteGeneratorRunData {
    std::size_t point_count;
    std::size_t segment_count;
    double parameter_step;
};

struct PiecewiseHermiteGeneratorResult {
    PiecewiseHermiteData piecewise_hermite;
    PiecewiseHermiteGeneratorRunData run_data;
};

[[nodiscard]] PiecewiseHermiteGeneratorResult piecewise_hermite_generator(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis,
    double spacing_world,
    double nonlocal_window_world,
    double max_curvature,
    double curvature_tolerance,
    double extension_step_world,
    int outer_iterations
);

}  // namespace autorigami
