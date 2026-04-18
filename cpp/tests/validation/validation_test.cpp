#include "autorigami/validation.h"

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

void test_straight_polyline_passes_curvature_validation() {
    const std::vector<autorigami::Vec3> points = {
        {.x = 0.0, .y = 0.0, .z = 0.0},
        {.x = 1.0, .y = 0.0, .z = 0.0},
        {.x = 2.0, .y = 0.0, .z = 0.0},
        {.x = 3.0, .y = 0.0, .z = 0.0},
    };

    const autorigami::ValidationReport report = autorigami::validate_polyline_constraints(
        points,
        1.0,
        0.5,
        0.5,
        1
    );

    expect(report.curvature.compliant_count == 4, "straight polyline should satisfy curvature");
    expect(report.curvature.total_count == 4, "straight polyline should report all samples");
    expect(report.curvature.ratio() == 1.0, "straight polyline should have full curvature ratio");
}

}  // namespace

int main() {
    try {
        test_straight_polyline_passes_curvature_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
