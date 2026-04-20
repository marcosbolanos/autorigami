#pragma once

#include "autorigami/generator.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/surface_point.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/vector3.h"

namespace autorigami {

[[nodiscard]] geometrycentral::surface::SurfacePoint initialize_surface_seed_point(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis
);

[[nodiscard]] bool projected_line_may_hit_triangle(
    const geometrycentral::Vector3& axis_origin,
    const geometrycentral::Vector3& axis_direction,
    const geometrycentral::Vector3& p0,
    const geometrycentral::Vector3& p1,
    const geometrycentral::Vector3& p2
);

}  // namespace autorigami
