#include "autorigami/generator/seed_point.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace autorigami {

// initialize a seed point on the surface of the mesh
geometrycentral::surface::SurfacePoint initialize_surface_seed_point(
    geometrycentral::surface::ManifoldSurfaceMesh& mesh,
    geometrycentral::surface::VertexPositionGeometry& geometry,
    const Vec3& axis
) {
    const geometrycentral::Vector3 axis_vector{axis.x, axis.y, axis.z};
    const double axis_norm2 = geometrycentral::norm2(axis_vector);
    if (axis_norm2 == 0.0) {
        throw std::invalid_argument("axis must be non-zero");
    }
    // get a vector of unit length in the axis direction
    const geometrycentral::Vector3 axis_unit = axis_vector / std::sqrt(axis_norm2);

    bool has_negative_candidate = false;
    bool has_positive_candidate = false;
    double best_negative_distance2 = std::numeric_limits<double>::infinity();
    double best_positive_distance2 = std::numeric_limits<double>::infinity();
    geometrycentral::surface::Vertex best_negative_vertex;
    geometrycentral::surface::Vertex best_positive_vertex;

    // loop thru all vertices in the mesh to find the one closest to the axis
    for (geometrycentral::surface::Vertex vertex : mesh.vertices()) {
        // project the vertice's coordinates along the chosen axis by dot product
        const geometrycentral::Vector3 position = geometry.inputVertexPositions[vertex];
        const double signed_axis_coordinate = geometrycentral::dot(position, axis_unit);
        const geometrycentral::Vector3 projected = axis_unit * signed_axis_coordinate;
        const double radial_distance2 = geometrycentral::norm2(position - projected);

        // save it if it's the closest to the axis we've seen, either on the positive or negative side
        if (signed_axis_coordinate <= 0.0) {
            if (!has_negative_candidate || radial_distance2 < best_negative_distance2) {
                has_negative_candidate = true;
                best_negative_distance2 = radial_distance2;
                best_negative_vertex = vertex;
            }
        } else {
            if (!has_positive_candidate || radial_distance2 < best_positive_distance2) {
                has_positive_candidate = true;
                best_positive_distance2 = radial_distance2;
                best_positive_vertex = vertex;
            }
        }
    }

    // points along the negative axis are used by default, by convention
    if (has_negative_candidate) {
        return geometrycentral::surface::SurfacePoint(best_negative_vertex);
    }
    if (has_positive_candidate) {
        return geometrycentral::surface::SurfacePoint(best_positive_vertex);
    }

    throw std::invalid_argument("mesh contains no vertices");
}

}  // namespace autorigami
