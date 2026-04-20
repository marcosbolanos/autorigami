#pragma once

#include "autorigami/vec3.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace autorigami {

[[nodiscard]] Vec3 initialize_surface_seed_point(
    const geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const Vec3& axis
);

}  // namespace autorigami
