#include "autorigami/splines.h"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(double actual, double expected, double tolerance, const char* message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

void expect_vec3_near(
    const autorigami::Vec3& actual,
    const autorigami::Vec3& expected,
    double tolerance,
    const char* message
) {
    expect_near(actual.x, expected.x, tolerance, message);
    expect_near(actual.y, expected.y, tolerance, message);
    expect_near(actual.z, expected.z, tolerance, message);
}

void test_polyline_construction_builds_one_segment_per_edge() {
    const std::vector<autorigami::Vec3> points = {
        {.x = 0.0, .y = 0.0, .z = 0.0},
        {.x = 1.0, .y = 0.0, .z = 0.0},
        {.x = 2.0, .y = 1.0, .z = 0.0},
        {.x = 3.0, .y = 1.0, .z = 0.0},
    };

    const autorigami::PiecewiseCubicHermiteSpline spline =
        autorigami::PiecewiseCubicHermiteSpline::from_polyline(
            points,
            autorigami::TangentPolicy::CatmullRomCentripetal
        );

    expect(spline.size() == 3, "expected one Hermite segment per polyline edge");

    const auto& segments = spline.segments();
    expect(segments[0].p0 == points[0], "segment 0 p0 should match point 0");
    expect(segments[0].p1 == points[1], "segment 0 p1 should match point 1");
    expect(segments[1].p0 == points[1], "segment 1 p0 should match point 1");
    expect(segments[1].p1 == points[2], "segment 1 p1 should match point 2");
    expect(segments[2].p0 == points[2], "segment 2 p0 should match point 2");
    expect(segments[2].p1 == points[3], "segment 2 p1 should match point 3");
}

void test_catmull_rom_centripetal_uses_expected_nonuniform_tangents() {
    const std::vector<autorigami::Vec3> points = {
        {.x = 0.0, .y = 0.0, .z = 0.0},
        {.x = 1.0, .y = 0.0, .z = 0.0},
        {.x = 2.0, .y = 2.0, .z = 0.0},
        {.x = 4.0, .y = 3.0, .z = 0.0},
    };

    const autorigami::PiecewiseCubicHermiteSpline spline =
        autorigami::PiecewiseCubicHermiteSpline::from_polyline(
            points,
            autorigami::TangentPolicy::CatmullRomCentripetal
        );

    const auto& segments = spline.segments();
    constexpr double tolerance = 1e-9;

    expect_near(segments[0].m0.x, 1.0, tolerance, "segment 0 m0.x mismatch");
    expect_near(segments[0].m0.y, 0.0, tolerance, "segment 0 m0.y mismatch");
    expect_near(segments[0].m1.x, 0.8672491406746515, tolerance, "segment 0 m1.x mismatch");
    expect_near(segments[0].m1.y, 0.5359894456510735, tolerance, "segment 0 m1.y mismatch");

    expect_near(segments[1].m0.x, 1.2968399455229909, tolerance, "segment 1 m0.x mismatch");
    expect_near(segments[1].m0.y, 0.8014911643017704, tolerance, "segment 1 m0.y mismatch");
    expect_near(segments[1].m1.x, 1.5, tolerance, "segment 1 m1.x mismatch");
    expect_near(segments[1].m1.y, 1.5, tolerance, "segment 1 m1.y mismatch");

    expect_near(segments[2].m0.x, 1.5, tolerance, "segment 2 m0.x mismatch");
    expect_near(segments[2].m0.y, 1.5, tolerance, "segment 2 m0.y mismatch");
    expect_near(segments[2].m1.x, 2.0, tolerance, "segment 2 m1.x mismatch");
    expect_near(segments[2].m1.y, 1.0, tolerance, "segment 2 m1.y mismatch");
}

void test_cubic_hermite_segment_converts_to_power_basis() {
    const autorigami::CubicHermiteSegment segment{
        .p0 = {.x = 1.0, .y = 2.0, .z = 3.0},
        .p1 = {.x = 4.0, .y = 6.0, .z = 8.0},
        .m0 = {.x = 0.5, .y = 1.5, .z = 2.5},
        .m1 = {.x = 1.0, .y = 2.0, .z = 3.0},
    };

    const autorigami::CubicPowerBasisSegment power_basis = segment.to_power_basis();

    constexpr double tolerance = 1e-12;
    expect_vec3_near(
        power_basis.a,
        {.x = -4.5, .y = -4.5, .z = -4.5},
        tolerance,
        "power basis a mismatch"
    );
    expect_vec3_near(
        power_basis.b,
        {.x = 7.0, .y = 7.0, .z = 7.0},
        tolerance,
        "power basis b mismatch"
    );
    expect_vec3_near(
        power_basis.c,
        {.x = 0.5, .y = 1.5, .z = 2.5},
        tolerance,
        "power basis c mismatch"
    );
    expect_vec3_near(
        power_basis.d,
        {.x = 1.0, .y = 2.0, .z = 3.0},
        tolerance,
        "power basis d mismatch"
    );
}

}  // namespace

int main() {
    try {
        test_polyline_construction_builds_one_segment_per_edge();
        test_catmull_rom_centripetal_uses_expected_nonuniform_tangents();
        test_cubic_hermite_segment_converts_to_power_basis();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
