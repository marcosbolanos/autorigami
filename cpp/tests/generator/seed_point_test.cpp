#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

#include "autorigami/generator.h"
#include "autorigami/generator/seed_point.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

namespace {

using geometrycentral::Vector3;
using geometrycentral::surface::Face;
using geometrycentral::surface::ManifoldSurfaceMesh;
using geometrycentral::surface::SurfacePoint;
using geometrycentral::surface::VertexPositionGeometry;

struct TestMesh {
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
};

[[nodiscard]] TestMesh make_tetrahedron() {
    const std::vector<Vector3> vertex_positions = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const std::vector<std::vector<std::size_t>> faces = {
        {0, 2, 1},
        {0, 1, 3},
        {1, 2, 3},
        {2, 0, 3},
    };
    auto [mesh, geometry] =
        geometrycentral::surface::makeManifoldSurfaceMeshAndGeometry(faces, vertex_positions);
    return TestMesh{
        .mesh = std::move(mesh),
        .geometry = std::move(geometry),
    };
}

void assert_vec3_close(const Vector3& actual, const Vector3& expected) {
    constexpr double tolerance = 1e-10;
    assert(std::abs(actual.x - expected.x) < tolerance);
    assert(std::abs(actual.y - expected.y) < tolerance);
    assert(std::abs(actual.z - expected.z) < tolerance);
}

void test_seed_point_prefers_negative_hit() {
    TestMesh tm = make_tetrahedron();
    const autorigami::GeneratorAxis axis{
        .origin = {.x = 0.1, .y = 0.1, .z = 0.4},
        .direction = {.x = 0.0, .y = 0.0, .z = 1.0},
    };

    const SurfacePoint seed =
        autorigami::initialize_surface_seed_point(*tm.mesh, *tm.geometry, axis);
    const Vector3 position = seed.interpolate(tm.geometry->inputVertexPositions);
    assert_vec3_close(position, Vector3{0.1, 0.1, 0.0});
}

void test_seed_point_falls_back_to_positive_hit() {
    TestMesh tm = make_tetrahedron();
    const autorigami::GeneratorAxis axis{
        .origin = {.x = 0.1, .y = 0.1, .z = -0.2},
        .direction = {.x = 0.0, .y = 0.0, .z = 1.0},
    };

    const SurfacePoint seed =
        autorigami::initialize_surface_seed_point(*tm.mesh, *tm.geometry, axis);
    const Vector3 position = seed.interpolate(tm.geometry->inputVertexPositions);
    assert_vec3_close(position, Vector3{0.1, 0.1, 0.0});
}

void test_seed_point_handles_x_axis_direction() {
    TestMesh tm = make_tetrahedron();
    const autorigami::GeneratorAxis axis{
        .origin = {.x = 0.4, .y = 0.1, .z = 0.1},
        .direction = {.x = 1.0, .y = 0.0, .z = 0.0},
    };

    const SurfacePoint seed =
        autorigami::initialize_surface_seed_point(*tm.mesh, *tm.geometry, axis);
    const Vector3 position = seed.interpolate(tm.geometry->inputVertexPositions);
    assert_vec3_close(position, Vector3{0.0, 0.1, 0.1});
}

void test_projected_line_may_hit_triangle_cases() {
    const Vector3 axis_origin{0.0, 0.0, 0.0};
    const Vector3 axis_direction{0.0, 0.0, 1.0};

    // Positive: projected triangle straddles projected axis line (x=0 in xy projection).
    assert(autorigami::projected_line_may_hit_triangle(
        axis_origin,
        axis_direction,
        Vector3{-1.0, 0.0, 0.2},
        Vector3{1.0, 0.0, 0.3},
        Vector3{0.0, 1.0, 0.1}
    ));

    // Positive: one projected vertex exactly on projected line should be kept.
    assert(autorigami::projected_line_may_hit_triangle(
        axis_origin,
        axis_direction,
        Vector3{0.0, 0.2, -0.1},
        Vector3{1.0, 0.3, 0.4},
        Vector3{1.2, -0.1, 0.5}
    ));

    // Negative: projected triangle strictly on one side of projected axis line.
    assert(!autorigami::projected_line_may_hit_triangle(
        axis_origin,
        axis_direction,
        Vector3{1.0, 0.0, 0.0},
        Vector3{2.0, 0.5, 0.1},
        Vector3{1.5, -0.5, -0.2}
    ));

    // Same logic for x-dominant axis (drop x, work in yz projection).
    const Vector3 x_axis_direction{1.0, 0.0, 0.0};
    assert(autorigami::projected_line_may_hit_triangle(
        axis_origin,
        x_axis_direction,
        Vector3{0.0, -1.0, 0.0},
        Vector3{0.2, 1.0, 0.0},
        Vector3{-0.1, 0.0, 0.5}
    ));
    assert(!autorigami::projected_line_may_hit_triangle(
        axis_origin,
        x_axis_direction,
        Vector3{0.0, 1.0, 0.2},
        Vector3{0.1, 2.0, -0.1},
        Vector3{-0.2, 1.5, 0.3}
    ));
}

}  // namespace

int main() {
    test_seed_point_prefers_negative_hit();
    test_seed_point_falls_back_to_positive_hit();
    test_seed_point_handles_x_axis_direction();
    test_projected_line_may_hit_triangle_cases();
    return 0;
}
