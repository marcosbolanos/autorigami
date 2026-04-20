#include "autorigami/generator/optimize_end_of_spline.h"

namespace autorigami {

PiecewiseHermiteData bootstrap_from_seed(
    const geometrycentral::Vector3& seed,
    const GeneratorAxis& axis
) {
    constexpr double step_size = 0.5;
    const geometrycentral::Vector3 tangent = axis.direction.normalizeCutoff() * step_size;

    // Minimal bootstrap: one seed point becomes a 2-point spline.
    return PiecewiseHermiteData{
        .points = {
            seed,
            seed + tangent,
        },
        .tangents = {
            tangent,
            tangent,
        },
    };
}

PiecewiseHermiteData optimize_end_of_spline(
    const PiecewiseHermiteData& current,
    const GeneratorAxis& axis,
    double max_curvature,
    double curvature_tolerance
) {
    (void)max_curvature;
    (void)curvature_tolerance;

    // TODO: run inner optimization over the tail DOFs:
    // - optimize last point
    // - optimize penultimate tangent
    // while checking curvature on both end segments.
    // This scaffold currently performs no optimization, only extension.

    PiecewiseHermiteData next = current;
    constexpr double step_size = 0.5;
    const geometrycentral::Vector3 tangent = axis.direction.normalizeCutoff() * step_size;

    if (next.points.size() < 2) {
        return bootstrap_from_seed(next.points.front(), axis);
    }

    next.points.push_back(next.points.back() + tangent);
    next.tangents.push_back(next.tangents.back());
    return next;
}

}  // namespace autorigami
