#include "autorigami/generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "autorigami/generator/optimize_end_of_spline.h"
#include "autorigami/generator/seed_point.h"
#include "geometrycentral/surface/halfedge_element_types.h"
#include "geometrycentral/surface/surface_point.h"
#include "geometrycentral/utilities/vector3.h"

namespace autorigami {

namespace {

constexpr std::size_t kSeedCandidatesPerPole = 8;
constexpr int kSeedEvaluationIterations = 48;
constexpr std::size_t kCoverageAngleBins = 48;
constexpr std::size_t kCoverageAxisBins = 64;
constexpr double kSmallEpsilon = 1e-12;
constexpr double kPi = 3.14159265358979323846;

struct AxisFrame {
    geometrycentral::Vector3 direction;
    geometrycentral::Vector3 u;
    geometrycentral::Vector3 v;
    double axis_min;
    double axis_max;
};

struct CandidateScore {
    double length;
    std::size_t coverage;
};

struct SeedCandidate {
    geometrycentral::surface::SurfacePoint seed;
    EndOfSplineStepResult initial_step;
    int evaluation_iterations;
    CandidateScore score;
};

[[nodiscard]] double polyline_length(const std::vector<geometrycentral::Vector3>& points) {
    if (points.size() < 2) {
        return 0.0;
    }
    double total = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        total += geometrycentral::norm(points[i] - points[i - 1]);
    }
    return total;
}

[[nodiscard]] AxisFrame build_axis_frame(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis
) {
    const geometrycentral::Vector3 direction = axis.direction.normalizeCutoff();
    if (geometrycentral::norm2(direction) < kSmallEpsilon) {
        throw std::invalid_argument("axis.direction must be non-zero");
    }

    geometrycentral::Vector3 helper = {1.0, 0.0, 0.0};
    if (std::abs(geometrycentral::dot(helper, direction)) > 0.9) {
        helper = {0.0, 1.0, 0.0};
    }
    const geometrycentral::Vector3 u = geometrycentral::cross(direction, helper).normalizeCutoff();
    const geometrycentral::Vector3 v = geometrycentral::cross(direction, u).normalizeCutoff();

    double axis_min = std::numeric_limits<double>::infinity();
    double axis_max = -std::numeric_limits<double>::infinity();
    for (geometrycentral::surface::Vertex vertex : mesh.vertices()) {
        const geometrycentral::Vector3 position = geometry.inputVertexPositions[vertex];
        const double coordinate = geometrycentral::dot(position - axis.origin, direction);
        axis_min = std::min(axis_min, coordinate);
        axis_max = std::max(axis_max, coordinate);
    }

    if (!std::isfinite(axis_min) || !std::isfinite(axis_max)) {
        throw std::runtime_error("mesh has no vertices for axis frame");
    }

    return AxisFrame{
        .direction = direction,
        .u = u,
        .v = v,
        .axis_min = axis_min,
        .axis_max = axis_max,
    };
}

[[nodiscard]] std::size_t estimate_coverage(
    const std::vector<geometrycentral::Vector3>& points,
    const GeneratorAxis& axis,
    const AxisFrame& frame
) {
    if (points.empty()) {
        return 0;
    }

    const double axis_span = std::max(frame.axis_max - frame.axis_min, kSmallEpsilon);
    std::unordered_set<std::uint64_t> occupied;
    occupied.reserve(points.size());

    for (const geometrycentral::Vector3& point : points) {
        const geometrycentral::Vector3 rel = point - axis.origin;
        const double axis_coord = geometrycentral::dot(rel, frame.direction);
        const double axis_normalized = std::clamp((axis_coord - frame.axis_min) / axis_span, 0.0, 1.0);
        std::size_t axis_bin = static_cast<std::size_t>(axis_normalized * kCoverageAxisBins);
        if (axis_bin >= kCoverageAxisBins) {
            axis_bin = kCoverageAxisBins - 1;
        }

        const geometrycentral::Vector3 radial = rel - frame.direction * axis_coord;
        double theta = std::atan2(geometrycentral::dot(radial, frame.v), geometrycentral::dot(radial, frame.u));
        if (theta < 0.0) {
            theta += 2.0 * kPi;
        }
        std::size_t angle_bin = static_cast<std::size_t>(theta / (2.0 * kPi) * kCoverageAngleBins);
        if (angle_bin >= kCoverageAngleBins) {
            angle_bin = kCoverageAngleBins - 1;
        }

        const std::uint64_t key =
            static_cast<std::uint64_t>(axis_bin) * static_cast<std::uint64_t>(kCoverageAngleBins) +
            static_cast<std::uint64_t>(angle_bin);
        occupied.insert(key);
    }

    return occupied.size();
}

[[nodiscard]] CandidateScore score_candidate(
    const std::vector<geometrycentral::Vector3>& points,
    const GeneratorAxis& axis,
    const AxisFrame& frame
) {
    return CandidateScore{
        .length = polyline_length(points),
        .coverage = estimate_coverage(points, axis, frame),
    };
}

[[nodiscard]] bool is_better_score(const CandidateScore& lhs, const CandidateScore& rhs) {
    if (lhs.length != rhs.length) {
        return lhs.length > rhs.length;
    }
    return lhs.coverage > rhs.coverage;
}

[[nodiscard]] std::vector<geometrycentral::surface::SurfacePoint> collect_seed_candidates(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis,
    const AxisFrame& frame
) {
    struct VertexProjection {
        geometrycentral::surface::Vertex vertex;
        double projection;
    };

    std::vector<VertexProjection> projections;
    projections.reserve(mesh.nVertices());
    for (geometrycentral::surface::Vertex vertex : mesh.vertices()) {
        const geometrycentral::Vector3 position = geometry.inputVertexPositions[vertex];
        const double projection = geometrycentral::dot(position - axis.origin, frame.direction);
        projections.push_back(VertexProjection{.vertex = vertex, .projection = projection});
    }

    std::sort(
        projections.begin(),
        projections.end(),
        [](const VertexProjection& a, const VertexProjection& b) { return a.projection < b.projection; }
    );

    std::vector<geometrycentral::surface::SurfacePoint> out;
    std::unordered_set<std::size_t> seen_vertices;

    auto add_vertex = [&](geometrycentral::surface::Vertex vertex) {
        const std::size_t id = static_cast<std::size_t>(vertex.getIndex());
        if (seen_vertices.contains(id)) {
            return;
        }
        seen_vertices.insert(id);
        out.push_back(geometrycentral::surface::SurfacePoint(vertex).inSomeFace());
    };

    const std::size_t count = projections.size();
    if (count > 0) {
        const std::size_t band = std::min<std::size_t>(count, std::max<std::size_t>(kSeedCandidatesPerPole, count / 12));
        const std::size_t per_pole = std::min<std::size_t>(kSeedCandidatesPerPole, band);

        for (std::size_t i = 0; i < per_pole; ++i) {
            const std::size_t offset = (per_pole <= 1) ? 0 : (i * (band - 1) / (per_pole - 1));
            add_vertex(projections[offset].vertex);
            add_vertex(projections[count - 1 - offset].vertex);
        }
    }

    if (count > 0) {
        add_vertex(projections[count / 2].vertex);
    }
    out.push_back(initialize_surface_seed_point(mesh, geometry, axis).inSomeFace());
    return out;
}

}  // namespace

PiecewiseHermiteGeneratorResult piecewise_hermite_generator(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis,
    double spacing_world,
    double nonlocal_window_world,
    double max_curvature,
    double curvature_tolerance,
    double extension_step_world,
    int outer_iterations
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

    const AxisFrame frame = build_axis_frame(mesh, geometry, axis);
    const std::vector<geometrycentral::surface::SurfacePoint> seed_points =
        collect_seed_candidates(mesh, geometry, axis, frame);

    if (seed_points.empty()) {
        throw std::runtime_error("no valid seed candidates were generated");
    }

    const int evaluation_iterations = std::min(outer_iterations, kSeedEvaluationIterations);
    std::optional<SeedCandidate> best_candidate;

    for (const geometrycentral::surface::SurfacePoint& seed_surface : seed_points) {
        EndOfSplineStepResult step;
        try {
            step = bootstrap_from_seed(
                mesh,
                geometry,
                seed_surface,
                axis,
                spacing_world,
                max_curvature,
                curvature_tolerance
            );
        } catch (const std::exception&) {
            continue;
        }

        int accepted_evaluation_iterations = 0;
        for (int iteration = 0; iteration < evaluation_iterations; ++iteration) {
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
            accepted_evaluation_iterations += 1;
        }

        const CandidateScore score = score_candidate(step.spline.points, axis, frame);
        const SeedCandidate candidate{
            .seed = seed_surface,
            .initial_step = step,
            .evaluation_iterations = accepted_evaluation_iterations,
            .score = score,
        };

        if (!best_candidate.has_value() || is_better_score(candidate.score, best_candidate->score)) {
            best_candidate = candidate;
        }
    }

    if (!best_candidate.has_value()) {
        throw std::runtime_error("failed to generate a feasible initial seed");
    }

    EndOfSplineStepResult step = best_candidate->initial_step;
    const int remaining_iterations = std::max(0, outer_iterations - best_candidate->evaluation_iterations);
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
