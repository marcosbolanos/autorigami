#include "autorigami/generator/seed_point.h"

namespace autorigami {

Vec3 initialize_surface_seed_point(
    const geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const Vec3& axis
) {
    (void)mesh;
    (void)geometry;
    (void)axis;

    // Placeholder: the real implementation will pick an extreme point along axis.
    return {.x = 0.0, .y = 0.0, .z = 0.0};
}

}  // namespace autorigami
