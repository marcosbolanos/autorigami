#pragma once

#include "autorigami/vec3.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/surface_point.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

namespace autorigami {

[[nodiscard]] geometrycentral::surface::SurfacePoint initialize_surface_seed_point(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const Vec3& axis
);

}  // namespace autorigami
