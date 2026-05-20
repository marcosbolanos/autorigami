#include "autorigami/generator.h"

#include <algorithm>
#include <stdexcept>

#include "autorigami/generator/initialization.h"
#include "autorigami/generator/optimize_end_of_spline.h"

namespace autorigami {

PiecewiseHermiteGeneratorResult piecewise_hermite_generator(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis,
    double spacing_world,
    double nonlocal_window_world,
    double max_curvature,
    double curvature_tolerance,
    double extension_step_world,
    int outer_iterations,
    const GeneratorInitializationOptions& initialization_options
) {
    if (outer_iterations < 0) {
        throw std::invalid_argument("outer_iterations must be >= 0");
    }
    if (spacing_world <= 0.0) {
        throw std::invalid_argument("spacing_world must be > 0");
    }
    if (nonlocal_window_world <= 0.0) {
        throw std::invalid_argument("nonlocal_window_world must be > 0");
    }

    const InitializationResult initialization = select_best_generator_initialization(
        mesh,
        geometry,
        axis,
        initialization_options,
        spacing_world,
        nonlocal_window_world,
        extension_step_world,
        max_curvature,
        curvature_tolerance,
        outer_iterations
    );

    EndOfSplineStepResult step = initialization.step;
    const int remaining_iterations = std::max(0, outer_iterations - initialization.evaluation_iterations);
    for (int iteration = 0; iteration < remaining_iterations; ++iteration) {
        const EndOfSplineStepResult next = optimize_end_of_spline(
            mesh,
            geometry,
            step.spline,
            step.state,
            axis,
            spacing_world,
            nonlocal_window_world,
            extension_step_world,
            max_curvature,
            curvature_tolerance
        );
        if (next.spline.points.size() == step.spline.points.size()) {
            break;
        }
        step = next;
    }

    const PiecewiseHermiteData& piecewise_hermite = step.spline;
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
