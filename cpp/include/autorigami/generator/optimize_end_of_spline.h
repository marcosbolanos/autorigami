#pragma once

#include "autorigami/generator.h"
#include <array>
#include "geometrycentral/surface/surface_point.h"

namespace autorigami {

struct SurfaceEndOptimizationState {
    geometrycentral::surface::SurfacePoint end_point_surface;
    std::array<double, 2> tangent_local;
};

struct EndOfSplineStepResult {
    PiecewiseHermiteData spline;
    SurfaceEndOptimizationState state;
};

[[nodiscard]] EndOfSplineStepResult bootstrap_from_seed(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const geometrycentral::surface::SurfacePoint& seed_surface,
    const GeneratorAxis& axis,
    double spacing = 2.6,
    double max_curvature = 1e6,
    double curvature_tolerance = 0.0
);

[[nodiscard]] EndOfSplineStepResult optimize_end_of_spline(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const PiecewiseHermiteData& current,
    const SurfaceEndOptimizationState& current_state,
    const GeneratorAxis& axis,
    double spacing_world,
    double nonlocal_window_world,
    double extension_step_world,
    double max_curvature,
    double curvature_tolerance
);

}  // namespace autorigami
