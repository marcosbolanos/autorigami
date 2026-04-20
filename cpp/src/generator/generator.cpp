#include "autorigami/generator.h"
#include "autorigami/generator/seed_point.h"

namespace autorigami {

PiecewiseHermiteGeneratorResult piecewise_hermite_generator(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis
) {
    const geometrycentral::surface::SurfacePoint seed_point_on_surface =
        initialize_surface_seed_point(mesh, geometry, axis);
    const geometrycentral::Vector3 seed_point =
        seed_point_on_surface.interpolate(geometry.inputVertexPositions);

    const PiecewiseHermiteData piecewise_hermite = {
        .points = {
            seed_point,
            {1.0, 0.4, -0.2},
            {2.1, -0.3, 0.6},
            {3.0, 0.1, 1.0},
        },
        .tangents = {
            {0.8, 0.2, -0.1},
            {0.9, -0.5, 0.4},
            {0.7, 0.6, 0.3},
            {0.5, 0.2, 0.4},
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
