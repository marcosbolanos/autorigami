#pragma once

#include <cstddef>
#include <vector>

#include "autorigami/vec3.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace autorigami {

struct PiecewiseHermiteData {
    std::vector<Vec3> points;
    std::vector<Vec3> tangents;
};

struct GeneratorAxis {
    Vec3 origin;
    Vec3 direction;
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
    const GeneratorAxis& axis
);

}  // namespace autorigami
