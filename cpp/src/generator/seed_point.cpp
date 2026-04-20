#include "autorigami/generator/seed_point.h"

#include "geometrycentral/surface/halfedge_element_types.h"
#include "geometrycentral/utilities/vector3.h"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace autorigami {

struct FaceHit {
    geometrycentral::surface::Face face;
    double t;
    geometrycentral::Vector3 barycentric;
};

struct Vec2 {
    double x;
    double y;
};

[[nodiscard]] int dominant_axis(const geometrycentral::Vector3& direction) {
    const double abs_x = std::abs(direction.x);
    const double abs_y = std::abs(direction.y);
    const double abs_z = std::abs(direction.z);
    if (abs_x >= abs_y && abs_x >= abs_z) {
        return 0;
    }
    if (abs_y >= abs_z) {
        return 1;
    }
    return 2;
}

[[nodiscard]] Vec2 project_to_2d(const geometrycentral::Vector3& value, int dropped_axis) {
    if (dropped_axis == 0) {
        return {.x = value.y, .y = value.z};
    }
    if (dropped_axis == 1) {
        return {.x = value.x, .y = value.z};
    }
    return {.x = value.x, .y = value.y};
}

[[nodiscard]] double oriented_2d_cross(
    const Vec2& direction,
    const Vec2& origin,
    const Vec2& point
) {
    const double rel_x = point.x - origin.x;
    const double rel_y = point.y - origin.y;
    return direction.x * rel_y - direction.y * rel_x;
}

// Check if an intersection is possible before intersecting
bool projected_line_may_hit_triangle(
    const geometrycentral::Vector3& axis_origin,
    const geometrycentral::Vector3& axis_direction,
    const geometrycentral::Vector3& p0,
    const geometrycentral::Vector3& p1,
    const geometrycentral::Vector3& p2
) {
    constexpr double epsilon = 1e-12;
    const int dropped_axis = dominant_axis(axis_direction);
    const Vec2 origin_2d = project_to_2d(axis_origin, dropped_axis);
    const Vec2 direction_2d = project_to_2d(axis_direction, dropped_axis);
    const Vec2 t0 = project_to_2d(p0, dropped_axis);
    const Vec2 t1 = project_to_2d(p1, dropped_axis);
    const Vec2 t2 = project_to_2d(p2, dropped_axis);

    const double side0 = oriented_2d_cross(direction_2d, origin_2d, t0);
    const double side1 = oriented_2d_cross(direction_2d, origin_2d, t1);
    const double side2 = oriented_2d_cross(direction_2d, origin_2d, t2);

    const bool has_positive = side0 > epsilon || side1 > epsilon || side2 > epsilon;
    const bool has_negative = side0 < -epsilon || side1 < -epsilon || side2 < -epsilon;

    // If projected triangle vertices are strictly on one side, projected line cannot cross it.
    // If signs mix (or touch near zero), keep the face as an exact-intersection candidate.
    return has_positive == has_negative;
}

// Gives us the point of intersection between a point and a face, or lack thereof
[[nodiscard]] std::optional<FaceHit> intersect_axis_with_face(
    geometrycentral::surface::Face face,
    const geometrycentral::surface::VertexPositionGeometry& geometry,
    const geometrycentral::Vector3& axis_origin,
    const geometrycentral::Vector3& axis_direction
) {
    constexpr double epsilon = 1e-12;

    const geometrycentral::surface::Halfedge he0 = face.halfedge();
    const geometrycentral::surface::Halfedge he1 = he0.next();
    const geometrycentral::surface::Halfedge he2 = he1.next();

    const geometrycentral::Vector3 p0 = geometry.inputVertexPositions[he0.vertex()];
    const geometrycentral::Vector3 p1 = geometry.inputVertexPositions[he1.vertex()];
    const geometrycentral::Vector3 p2 = geometry.inputVertexPositions[he2.vertex()];

    // Return early if an intersection isn't possible
    if (!projected_line_may_hit_triangle(axis_origin, axis_direction, p0, p1, p2)) {
        return std::nullopt;
    }

    const geometrycentral::Vector3 edge1 = p1 - p0;
    const geometrycentral::Vector3 edge2 = p2 - p0;
    const geometrycentral::Vector3 pvec = geometrycentral::cross(axis_direction, edge2);

    // Möller-Trumbore style logic for intersection
    const double determinant = geometrycentral::dot(edge1, pvec);
    if (std::abs(determinant) < epsilon) {
        return std::nullopt;
    }
    const double inv_determinant = 1.0 / determinant;
    const geometrycentral::Vector3 tvec = axis_origin - p0;
    const double bary_v = geometrycentral::dot(tvec, pvec) * inv_determinant;
    if (bary_v < -epsilon || bary_v > 1.0 + epsilon) {
        return std::nullopt;
    }
    const geometrycentral::Vector3 qvec = geometrycentral::cross(tvec, edge1);
    const double bary_w = geometrycentral::dot(axis_direction, qvec) * inv_determinant;
    if (bary_w < -epsilon || bary_v + bary_w > 1.0 + epsilon) {
        return std::nullopt;
    }
    const double axis_t = geometrycentral::dot(edge2, qvec) * inv_determinant;
    const double bary_u = 1.0 - bary_v - bary_w;

    return FaceHit{
        .face = face,
        .t = axis_t,
        .barycentric = {bary_u, bary_v, bary_w},
    };
}

// Here, we intersect our axis with every single face.
// We return the intersection point that's furthest along the given origin axis
geometrycentral::surface::SurfacePoint initialize_surface_seed_point(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const GeneratorAxis& axis
) {
    const geometrycentral::Vector3 axis_origin{
        axis.origin.x,
        axis.origin.y,
        axis.origin.z,
    };
    const geometrycentral::Vector3 axis_direction{
        axis.direction.x,
        axis.direction.y,
        axis.direction.z,
    };

    constexpr double epsilon = 1e-12;
    const double direction_norm2 = geometrycentral::norm2(axis_direction);
    if (direction_norm2 <= epsilon) {
        throw std::invalid_argument("axis.direction must be non-zero");
    }

    std::optional<FaceHit> best_hit;

    for (geometrycentral::surface::Face face : mesh.faces()) {
        const std::optional<FaceHit> hit =
            intersect_axis_with_face(face, geometry, axis_origin, axis_direction);
        if (!hit.has_value()) {
            continue;
        }
           
        if (!best_hit.has_value() || std::abs(hit->t) < std::abs(best_hit->t)) {
            best_hit = hit;
        }
    }

    if (best_hit.has_value()) {
        return geometrycentral::surface::SurfacePoint(
            best_hit->face,
            best_hit->barycentric
        );
    }

    throw std::invalid_argument("axis line does not intersect any mesh face");
}

}  // namespace autorigami
