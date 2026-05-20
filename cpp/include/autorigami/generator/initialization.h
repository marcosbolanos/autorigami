#pragma once

#include "autorigami/generator/optimize_end_of_spline.h"

namespace autorigami {

struct InitializationResult {
    EndOfSplineStepResult step;
    int evaluation_iterations;
};

[[nodiscard]] InitializationResult select_best_generator_initialization(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis,
    const GeneratorInitializationOptions& options,
    double spacing_world,
    double nonlocal_window_world,
    double extension_step_world,
    double max_curvature,
    double curvature_tolerance,
    int outer_iterations
);

}  // namespace autorigami
