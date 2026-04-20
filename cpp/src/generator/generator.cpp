#include "autorigami/generator.h"
#include "autorigami/generator/seed_point.h"

namespace autorigami {

PiecewiseHermiteGeneratorResult piecewise_hermite_generator(
    const geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const Vec3& axis
) {
    const Vec3 seed_point = initialize_surface_seed_point(mesh, geometry, axis);

    const PiecewiseHermiteData piecewise_hermite = {
        .points = {
            seed_point,
            {.x = 1.0, .y = 0.4, .z = -0.2},
            {.x = 2.1, .y = -0.3, .z = 0.6},
            {.x = 3.0, .y = 0.1, .z = 1.0},
        },
        .tangents = {
            {.x = 0.8, .y = 0.2, .z = -0.1},
            {.x = 0.9, .y = -0.5, .z = 0.4},
            {.x = 0.7, .y = 0.6, .z = 0.3},
            {.x = 0.5, .y = 0.2, .z = 0.4},
        },
    };

    return PiecewiseHermiteGeneratorResult{
        .piecewise_hermite = piecewise_hermite,
        .run_data =
            {
                .point_count = piecewise_hermite.points.size(),
                .segment_count = piecewise_hermite.points.size() - 1,
                .parameter_step = 1.0,
            },
    };
}

}  // namespace autorigami
