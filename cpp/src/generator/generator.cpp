#include "autorigami/generator.h"
#include "autorigami/generator/optimize_end_of_spline.h"
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
    PiecewiseHermiteData piecewise_hermite = bootstrap_from_seed(seed_point, axis);
    constexpr int outer_iterations = 4;
    constexpr double max_curvature = 1e6;
    constexpr double curvature_tolerance = 0.0;
    for (int iteration = 0; iteration < outer_iterations; ++iteration) {
        piecewise_hermite = optimize_end_of_spline(
            piecewise_hermite,
            axis,
            max_curvature,
            curvature_tolerance
        );
    }

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
