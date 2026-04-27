#include "autorigami/generator/optimize_end_of_spline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "geometrycentral/utilities/vector3.h"

namespace autorigami {

namespace {

constexpr double kSmallLengthEpsilon = 1e-12;

[[nodiscard]] geometrycentral::surface::Face point_face(
    const geometrycentral::surface::SurfacePoint& point
) {
    if (point.type != geometrycentral::surface::SurfacePointType::Face) {
        throw std::invalid_argument("end_point_surface must be a face SurfacePoint");
    }
    return point.face;
}

[[nodiscard]] geometrycentral::Vector3 surface_point_position(
    const geometrycentral::surface::SurfacePoint& point,
    const geometrycentral::surface::VertexPositionGeometry& geometry
) {
    return point.interpolate(geometry.inputVertexPositions);
}

[[nodiscard]] std::array<geometrycentral::surface::Vertex, 3> face_vertices(
    geometrycentral::surface::Face face
) {
    const geometrycentral::surface::Halfedge he0 = face.halfedge();
    const geometrycentral::surface::Halfedge he1 = he0.next();
    const geometrycentral::surface::Halfedge he2 = he1.next();
    return {he0.vertex(), he1.vertex(), he2.vertex()};
}

[[nodiscard]] geometrycentral::Vector3 face_edge_u(
    geometrycentral::surface::Face face,
    const geometrycentral::surface::VertexPositionGeometry& geometry
) {
    const auto verts = face_vertices(face);
    return geometry.inputVertexPositions[verts[1]] - geometry.inputVertexPositions[verts[0]];
}

[[nodiscard]] geometrycentral::Vector3 face_edge_v(
    geometrycentral::surface::Face face,
    const geometrycentral::surface::VertexPositionGeometry& geometry
) {
    const auto verts = face_vertices(face);
    return geometry.inputVertexPositions[verts[2]] - geometry.inputVertexPositions[verts[0]];
}

[[nodiscard]] std::array<double, 2> project_vector_to_face_basis(
    geometrycentral::surface::Face face,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const geometrycentral::Vector3& vector
) {
    const geometrycentral::Vector3 u = face_edge_u(face, geometry);
    const geometrycentral::Vector3 v = face_edge_v(face, geometry);

    const double uu = geometrycentral::dot(u, u);
    const double uv = geometrycentral::dot(u, v);
    const double vv = geometrycentral::dot(v, v);
    const double rhs_u = geometrycentral::dot(u, vector);
    const double rhs_v = geometrycentral::dot(v, vector);
    const double det = uu * vv - uv * uv;
    if (std::abs(det) < 1e-14) {
        return {0.0, 0.0};
    }
    return {
        (rhs_u * vv - rhs_v * uv) / det,
        (rhs_v * uu - rhs_u * uv) / det,
    };
}

[[nodiscard]] geometrycentral::Vector3 lift_local_to_face_tangent(
    geometrycentral::surface::Face face,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const std::array<double, 2>& local
) {
    const geometrycentral::Vector3 u = face_edge_u(face, geometry);
    const geometrycentral::Vector3 v = face_edge_v(face, geometry);
    return u * local[0] + v * local[1];
}

[[nodiscard]] std::array<double, 3> barycentric_step_from_local(
    const std::array<double, 2>& delta_local
) {
    return {
        -delta_local[0] - delta_local[1],
        delta_local[0],
        delta_local[1],
    };
}

struct ClosestFacePoint {
    geometrycentral::surface::Face face;
    geometrycentral::Vector3 barycentric;
    double squared_distance;
};

[[nodiscard]] ClosestFacePoint closest_point_on_triangle(
    geometrycentral::surface::Face face,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const geometrycentral::Vector3& query
) {
    const auto verts = face_vertices(face);
    const geometrycentral::Vector3 a = geometry.inputVertexPositions[verts[0]];
    const geometrycentral::Vector3 b = geometry.inputVertexPositions[verts[1]];
    const geometrycentral::Vector3 c = geometry.inputVertexPositions[verts[2]];

    const geometrycentral::Vector3 ab = b - a;
    const geometrycentral::Vector3 ac = c - a;
    const geometrycentral::Vector3 ap = query - a;
    const double d1 = geometrycentral::dot(ab, ap);
    const double d2 = geometrycentral::dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return {.face = face, .barycentric = {1.0, 0.0, 0.0}, .squared_distance = geometrycentral::norm2(query - a)};
    }

    const geometrycentral::Vector3 bp = query - b;
    const double d3 = geometrycentral::dot(ab, bp);
    const double d4 = geometrycentral::dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return {.face = face, .barycentric = {0.0, 1.0, 0.0}, .squared_distance = geometrycentral::norm2(query - b)};
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        const geometrycentral::Vector3 point = a + ab * v;
        return {.face = face, .barycentric = {1.0 - v, v, 0.0}, .squared_distance = geometrycentral::norm2(query - point)};
    }

    const geometrycentral::Vector3 cp = query - c;
    const double d5 = geometrycentral::dot(ab, cp);
    const double d6 = geometrycentral::dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return {.face = face, .barycentric = {0.0, 0.0, 1.0}, .squared_distance = geometrycentral::norm2(query - c)};
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        const geometrycentral::Vector3 point = a + ac * w;
        return {.face = face, .barycentric = {1.0 - w, 0.0, w}, .squared_distance = geometrycentral::norm2(query - point)};
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const geometrycentral::Vector3 bc = c - b;
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const geometrycentral::Vector3 point = b + bc * w;
        return {.face = face, .barycentric = {0.0, 1.0 - w, w}, .squared_distance = geometrycentral::norm2(query - point)};
    }

    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom;
    const double w = vc * denom;
    const double u = 1.0 - v - w;
    const geometrycentral::Vector3 point = a * u + b * v + c * w;
    return {.face = face, .barycentric = {u, v, w}, .squared_distance = geometrycentral::norm2(query - point)};
}

[[nodiscard]] geometrycentral::surface::SurfacePoint closest_surface_point_on_mesh(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const geometrycentral::Vector3& query
) {
    bool has_best = false;
    ClosestFacePoint best{};
    for (geometrycentral::surface::Face face : mesh.faces()) {
        const ClosestFacePoint candidate = closest_point_on_triangle(face, geometry, query);
        if (!has_best || candidate.squared_distance < best.squared_distance) {
            best = candidate;
            has_best = true;
        }
    }
    if (!has_best) {
        throw std::runtime_error("mesh has no faces");
    }
    return geometrycentral::surface::SurfacePoint(best.face, best.barycentric);
}

[[nodiscard]] geometrycentral::surface::SurfacePoint take_face_step(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const geometrycentral::surface::SurfacePoint& start,
    const std::array<double, 2>& delta_local
) {
    const geometrycentral::Vector3 current_position = surface_point_position(start, geometry);
    const geometrycentral::surface::Face start_face = point_face(start.inSomeFace());
    const geometrycentral::Vector3 world_step =
        lift_local_to_face_tangent(start_face, geometry, delta_local);
    const geometrycentral::Vector3 target_position = current_position + world_step;
    return closest_surface_point_on_mesh(mesh, geometry, target_position);
}

[[nodiscard]] std::vector<double> cumulative_chord_lengths(
    const std::vector<geometrycentral::Vector3>& points
) {
    std::vector<double> cumulative(points.size(), 0.0);
    for (std::size_t i = 1; i < points.size(); ++i) {
        const double step = geometrycentral::norm(points[i] - points[i - 1]);
        cumulative[i] = cumulative[i - 1] + step;
    }
    return cumulative;
}

[[nodiscard]] std::vector<geometrycentral::Vector3> tangents_from_polyline(
    const std::vector<geometrycentral::Vector3>& points
) {
    if (points.size() < 2) {
        return {};
    }

    std::vector<geometrycentral::Vector3> tangents(points.size(), geometrycentral::Vector3{0.0, 0.0, 0.0});
    tangents[0] = (points[1] - points[0]) * 0.25;
    tangents[points.size() - 1] = (points[points.size() - 1] - points[points.size() - 2]) * 0.25;
    for (std::size_t i = 1; i + 1 < points.size(); ++i) {
        const geometrycentral::Vector3 prev_edge = points[i] - points[i - 1];
        const geometrycentral::Vector3 next_edge = points[i + 1] - points[i];
        const double prev_len = geometrycentral::norm(prev_edge);
        const double next_len = geometrycentral::norm(next_edge);
        if (prev_len < kSmallLengthEpsilon || next_len < kSmallLengthEpsilon) {
            tangents[i] = geometrycentral::Vector3{0.0, 0.0, 0.0};
            continue;
        }

        geometrycentral::Vector3 bisector = prev_edge / prev_len + next_edge / next_len;
        if (geometrycentral::norm2(bisector) < kSmallLengthEpsilon) {
            tangents[i] = geometrycentral::Vector3{0.0, 0.0, 0.0};
            continue;
        }
        bisector = bisector.normalizeCutoff();
        const double tangent_len = 0.25 * std::min(prev_len, next_len);
        tangents[i] = bisector * tangent_len;
    }
    return tangents;
}

[[nodiscard]] double discrete_tip_curvature(
    const std::vector<geometrycentral::Vector3>& points
) {
    if (points.size() < 3) {
        return 0.0;
    }
    const geometrycentral::Vector3& p_prev = points[points.size() - 3];
    const geometrycentral::Vector3& p_curr = points[points.size() - 2];
    const geometrycentral::Vector3& p_next = points[points.size() - 1];

    const geometrycentral::Vector3 e0 = p_curr - p_prev;
    const geometrycentral::Vector3 e1 = p_next - p_curr;
    const double l0 = geometrycentral::norm(e0);
    const double l1 = geometrycentral::norm(e1);
    if (l0 < kSmallLengthEpsilon || l1 < kSmallLengthEpsilon) {
        return std::numeric_limits<double>::infinity();
    }

    const geometrycentral::Vector3 t0 = e0 / l0;
    const geometrycentral::Vector3 t1 = e1 / l1;
    const double ds = 0.5 * (l0 + l1);
    if (ds < kSmallLengthEpsilon) {
        return std::numeric_limits<double>::infinity();
    }
    return geometrycentral::norm(t1 - t0) / ds;
}

[[nodiscard]] geometrycentral::Vector3 wrap_biased_initial_direction(
    const geometrycentral::Vector3& seed_xyz,
    const GeneratorAxis& axis,
    const geometrycentral::Vector3& axis_direction
) {
    const geometrycentral::Vector3 rel = seed_xyz - axis.origin;
    const double axial = geometrycentral::dot(rel, axis_direction);
    const geometrycentral::Vector3 radial = rel - axis_direction * axial;
    geometrycentral::Vector3 wrap = geometrycentral::cross(axis_direction, radial);
    if (geometrycentral::norm2(wrap) < kSmallLengthEpsilon) {
        wrap = axis_direction;
    } else {
        wrap = wrap.normalizeCutoff();
    }

    const geometrycentral::Vector3 blended = wrap * 0.9 + axis_direction * 0.1;
    if (geometrycentral::norm2(blended) < kSmallLengthEpsilon) {
        return axis_direction;
    }
    return blended.normalizeCutoff();
}

[[nodiscard]] double segment_segment_distance_squared(
    const geometrycentral::Vector3& p0,
    const geometrycentral::Vector3& p1,
    const geometrycentral::Vector3& q0,
    const geometrycentral::Vector3& q1
) {
    const geometrycentral::Vector3 u = p1 - p0;
    const geometrycentral::Vector3 v = q1 - q0;
    const geometrycentral::Vector3 w = p0 - q0;

    const double a = geometrycentral::dot(u, u);
    const double b = geometrycentral::dot(u, v);
    const double c = geometrycentral::dot(v, v);
    const double d = geometrycentral::dot(u, w);
    const double e = geometrycentral::dot(v, w);

    const double denom = a * c - b * b;
    double s_num = 0.0;
    double s_den = denom;
    double t_num = 0.0;
    double t_den = denom;

    if (denom < kSmallLengthEpsilon) {
        s_num = 0.0;
        s_den = 1.0;
        t_num = e;
        t_den = c;
    } else {
        s_num = b * e - c * d;
        t_num = a * e - b * d;

        if (s_num < 0.0) {
            s_num = 0.0;
            t_num = e;
            t_den = c;
        } else if (s_num > s_den) {
            s_num = s_den;
            t_num = e + b;
            t_den = c;
        }
    }

    if (t_num < 0.0) {
        t_num = 0.0;
        if (-d < 0.0) {
            s_num = 0.0;
        } else if (-d > a) {
            s_num = s_den;
        } else {
            s_num = -d;
            s_den = a;
        }
    } else if (t_num > t_den) {
        t_num = t_den;
        if ((-d + b) < 0.0) {
            s_num = 0.0;
        } else if ((-d + b) > a) {
            s_num = s_den;
        } else {
            s_num = -d + b;
            s_den = a;
        }
    }

    const double s = (std::abs(s_num) < kSmallLengthEpsilon ? 0.0 : s_num / s_den);
    const double t = (std::abs(t_num) < kSmallLengthEpsilon ? 0.0 : t_num / t_den);

    const geometrycentral::Vector3 delta = w + u * s - v * t;
    return geometrycentral::dot(delta, delta);
}

[[nodiscard]] double segment_arc_gap(
    double a0,
    double a1,
    double b0,
    double b1
) {
    if (a1 <= b0) {
        return b0 - a1;
    }
    if (b1 <= a0) {
        return a0 - b1;
    }
    return 0.0;
}

[[nodiscard]] bool violates_minimum_separation(
    const std::vector<geometrycentral::Vector3>& points,
    const std::vector<double>& cumulative_lengths,
    std::size_t first_new_segment_index,
    double minimum_separation,
    double nonlocal_window
) {
    if (points.size() < 4) {
        return false;
    }

    const double minimum_separation_sq = minimum_separation * minimum_separation;
    const std::size_t segment_count = points.size() - 1;
    for (std::size_t i = first_new_segment_index; i < segment_count; ++i) {
        const geometrycentral::Vector3 a0 = points[i];
        const geometrycentral::Vector3 a1 = points[i + 1];
        const double arc_a0 = cumulative_lengths[i];
        const double arc_a1 = cumulative_lengths[i + 1];

        for (std::size_t j = 0; j < i; ++j) {
            const double arc_b0 = cumulative_lengths[j];
            const double arc_b1 = cumulative_lengths[j + 1];
            if (segment_arc_gap(arc_a0, arc_a1, arc_b0, arc_b1) < nonlocal_window) {
                continue;
            }

            const geometrycentral::Vector3 b0 = points[j];
            const geometrycentral::Vector3 b1 = points[j + 1];
            const double distance_sq = segment_segment_distance_squared(a0, a1, b0, b1);
            if (distance_sq < minimum_separation_sq) {
                return true;
            }
        }
    }

    return false;
}

}  // namespace

EndOfSplineStepResult bootstrap_from_seed(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const geometrycentral::surface::SurfacePoint& seed_surface,
    const GeneratorAxis& axis,
    double spacing,
    double max_curvature,
    double curvature_tolerance
) {
    static_cast<void>(max_curvature);
    static_cast<void>(curvature_tolerance);

    if (spacing <= 0.0) {
        throw std::invalid_argument("spacing must be > 0");
    }
    const geometrycentral::Vector3 axis_direction = axis.direction.normalizeCutoff();
    if (geometrycentral::norm2(axis_direction) == 0.0) {
        throw std::invalid_argument("axis.direction must be non-zero");
    }

    const geometrycentral::surface::Face seed_face = point_face(seed_surface.inSomeFace());
    const geometrycentral::Vector3 seed_xyz = surface_point_position(seed_surface, geometry);
    const geometrycentral::Vector3 initial_direction = wrap_biased_initial_direction(
        seed_xyz,
        axis,
        axis_direction
    );
    const std::array<double, 2> axis_local =
        project_vector_to_face_basis(seed_face, geometry, initial_direction * spacing);

    constexpr int max_attempts = 16;
    constexpr double jitter_scale = 0.03;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(-jitter_scale, jitter_scale);

    geometrycentral::surface::SurfacePoint chosen_end = take_face_step(
        mesh,
        geometry,
        seed_surface,
        axis_local
    );
    std::array<double, 2> chosen_tangent_local = axis_local;
    geometrycentral::Vector3 chosen_tangent = lift_local_to_face_tangent(seed_face, geometry, axis_local);
    geometrycentral::Vector3 chosen_endpoint = seed_xyz;
    bool found_feasible = false;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::array<double, 2> trial_local = axis_local;
        if (attempt > 0) {
            trial_local[0] += jitter(rng);
            trial_local[1] += jitter(rng);
        }
        const geometrycentral::Vector3 trial_step_world =
            lift_local_to_face_tangent(seed_face, geometry, trial_local);
        if (geometrycentral::norm(trial_step_world) < kSmallLengthEpsilon) {
            continue;
        }

        const geometrycentral::surface::SurfacePoint trial_end = take_face_step(
            mesh,
            geometry,
            seed_surface,
            trial_local
        ).inSomeFace();
        const geometrycentral::Vector3 trial_tangent =
            lift_local_to_face_tangent(point_face(trial_end), geometry, trial_local);
        const geometrycentral::Vector3 trial_endpoint = surface_point_position(trial_end, geometry);
        const double trial_segment_length = geometrycentral::norm(trial_endpoint - seed_xyz);
        if (trial_segment_length < kSmallLengthEpsilon) {
            continue;
        }

        chosen_end = trial_end;
        chosen_tangent_local = trial_local;
        chosen_tangent = trial_tangent;
        chosen_endpoint = trial_endpoint;
        found_feasible = true;
        break;
    }

    if (!found_feasible) {
        double best_dist2 = -1.0;
        geometrycentral::Vector3 farthest_vertex_position = seed_xyz;
        for (geometrycentral::surface::Vertex vertex : mesh.vertices()) {
            const geometrycentral::Vector3 candidate = geometry.inputVertexPositions[vertex];
            const double dist2 = geometrycentral::norm2(candidate - seed_xyz);
            if (dist2 > best_dist2) {
                best_dist2 = dist2;
                farthest_vertex_position = candidate;
            }
        }
        if (best_dist2 <= kSmallLengthEpsilon) {
            throw std::runtime_error("bootstrap failed to find a feasible non-degenerate first segment");
        }

        chosen_end = closest_surface_point_on_mesh(mesh, geometry, farthest_vertex_position).inSomeFace();
        chosen_endpoint = surface_point_position(chosen_end, geometry);
        const geometrycentral::Vector3 fallback_tangent = chosen_endpoint - seed_xyz;
        chosen_tangent = fallback_tangent;
        chosen_tangent_local = project_vector_to_face_basis(seed_face, geometry, fallback_tangent);
    }

    return EndOfSplineStepResult{
        .spline =
            PiecewiseHermiteData{
                .points = {
                    seed_xyz,
                    chosen_endpoint,
                },
                .tangents = {
                    chosen_tangent,
                    chosen_tangent,
                },
            },
        .state =
            SurfaceEndOptimizationState{
                .end_point_surface = chosen_end,
                .tangent_local = chosen_tangent_local,
            },
    };
}

EndOfSplineStepResult optimize_end_of_spline(
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
) {
    if (current.points.empty()) {
        throw std::invalid_argument("current spline must have at least one point");
    }

    const geometrycentral::surface::Face current_face = point_face(current_state.end_point_surface.inSomeFace());
    const geometrycentral::Vector3 axis_direction = axis.direction.normalizeCutoff();
    if (geometrycentral::norm2(axis_direction) == 0.0) {
        throw std::invalid_argument("axis.direction must be non-zero");
    }

    if (extension_step_world <= 0.0) {
        throw std::invalid_argument("extension_step_world must be > 0");
    }
    if (spacing_world <= 0.0) {
        throw std::invalid_argument("spacing_world must be > 0");
    }
    if (nonlocal_window_world <= 0.0) {
        throw std::invalid_argument("nonlocal_window_world must be > 0");
    }
    if (max_curvature < 0.0) {
        throw std::invalid_argument("max_curvature must be >= 0");
    }
    if (curvature_tolerance < 0.0) {
        throw std::invalid_argument("curvature_tolerance must be >= 0");
    }
    geometrycentral::Vector3 preferred_direction = axis_direction;
    if (current.points.size() >= 2) {
        const geometrycentral::Vector3 last_segment =
            current.points.back() - current.points[current.points.size() - 2];
        if (geometrycentral::norm2(last_segment) > kSmallLengthEpsilon) {
            const geometrycentral::Vector3 last_direction = last_segment.normalizeCutoff();
            preferred_direction = (last_direction * 0.9 + axis_direction * 0.1).normalizeCutoff();
        }
    }
    const std::array<double, 2> preferred_step_local = project_vector_to_face_basis(
        current_face,
        geometry,
        preferred_direction * extension_step_world
    );

    auto trial_step = [&](double scale) -> std::optional<EndOfSplineStepResult> {
        const std::array<double, 2> endpoint_step_local = {
            preferred_step_local[0] * scale,
            preferred_step_local[1] * scale,
        };
        const geometrycentral::Vector3 trial_step_world =
            lift_local_to_face_tangent(current_face, geometry, endpoint_step_local);
        if (geometrycentral::norm(trial_step_world) < kSmallLengthEpsilon) {
            return std::nullopt;
        }

        const geometrycentral::surface::SurfacePoint next_surface = take_face_step(
            mesh,
            geometry,
            current_state.end_point_surface,
            endpoint_step_local
        ).inSomeFace();
        const geometrycentral::Vector3 next_point = surface_point_position(next_surface, geometry);
        const double arc_step = geometrycentral::norm(next_point - current.points.back());
        if (arc_step < kSmallLengthEpsilon) {
            return std::nullopt;
        }

        const std::vector<double> existing_arclength = cumulative_chord_lengths(current.points);

        PiecewiseHermiteData trial = current;
        std::vector<double> trial_arclength = existing_arclength;
        trial.points.push_back(next_point);
        trial_arclength.push_back(existing_arclength.back() + arc_step);

        const std::size_t first_new_segment_index = current.points.size() - 1;
        if (violates_minimum_separation(
                trial.points,
                trial_arclength,
                first_new_segment_index,
                spacing_world,
                nonlocal_window_world
            )) {
            return std::nullopt;
        }

        const double tip_curvature = discrete_tip_curvature(trial.points);
        if (!std::isfinite(tip_curvature) ||
            tip_curvature > (max_curvature + curvature_tolerance)) {
            return std::nullopt;
        }

        trial.tangents = tangents_from_polyline(trial.points);

        return EndOfSplineStepResult{
            .spline = trial,
            .state =
                SurfaceEndOptimizationState{
                    .end_point_surface = next_surface,
                    .tangent_local = endpoint_step_local,
                },
        };
    };

    constexpr int max_backtracks = 10;
    double scale = 1.0;
    for (int k = 0; k < max_backtracks; ++k) {
        const std::optional<EndOfSplineStepResult> result = trial_step(scale);
        if (result.has_value()) {
            return *result;
        }
        scale *= 0.5;
    }

    return EndOfSplineStepResult{
        .spline = current,
        .state = current_state,
    };
}

}  // namespace autorigami
