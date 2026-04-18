#include "autorigami/splines.h"
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

autorigami::CubicPowerBasisSegment valid_curve_one() {
    return {
        .a = {.x = 0.0, .y = 0.0, .z = 0.0},
        .b = {.x = 0.0, .y = 1.0, .z = 0.0},
        .c = {.x = 1.0, .y = -0.5, .z = 0.0},
        .d = {.x = 0.0, .y = 0.0625, .z = 0.0},
    };
}

autorigami::CubicPowerBasisSegment valid_curve_two() {
    return {
        .a = {.x = 0.0, .y = 0.0, .z = 0.0},
        .b = {.x = 0.0, .y = 0.5, .z = 0.0},
        .c = {.x = 1.0, .y = -0.75, .z = 0.0},
        .d = {.x = 0.0, .y = 0.28125, .z = 0.0},
    };
}

autorigami::CubicPowerBasisSegment invalid_curve() {
    return {
        .a = {.x = 0.0, .y = 0.0, .z = 0.0},
        .b = {.x = 0.0, .y = 3.0, .z = 0.0},
        .c = {.x = 1.0, .y = -3.0, .z = 0.0},
        .d = {.x = 0.0, .y = 0.75, .z = 0.0},
    };
}

autorigami::PiecewiseCubicHermiteSpline valid_piecewise_curve() {
    const std::vector<autorigami::CubicHermiteSegment> segments = {
        {
            .p0 = {.x = 0.0, .y = 0.0625, .z = 0.0},
            .p1 = {.x = 1.0, .y = 0.5625, .z = 0.0},
            .m0 = {.x = 1.0, .y = -0.5, .z = 0.0},
            .m1 = {.x = 1.0, .y = 1.5, .z = 0.0},
        },
        {
            .p0 = {.x = 1.0, .y = 0.28125, .z = 0.0},
            .p1 = {.x = 2.0, .y = 0.03125, .z = 0.0},
            .m0 = {.x = 1.0, .y = -0.75, .z = 0.0},
            .m1 = {.x = 1.0, .y = 0.25, .z = 0.0},
        },
    };

    return autorigami::PiecewiseCubicHermiteSpline(segments);
}

autorigami::PiecewiseCubicHermiteSpline invalid_piecewise_curve() {
    const std::vector<autorigami::CubicHermiteSegment> segments = {
        {
            .p0 = {.x = 0.0, .y = 0.0625, .z = 0.0},
            .p1 = {.x = 1.0, .y = 0.5625, .z = 0.0},
            .m0 = {.x = 1.0, .y = -0.5, .z = 0.0},
            .m1 = {.x = 1.0, .y = 1.5, .z = 0.0},
        },
        {
            .p0 = {.x = 1.0, .y = 0.75, .z = 0.0},
            .p1 = {.x = 2.0, .y = 0.75, .z = 0.0},
            .m0 = {.x = 1.0, .y = -3.0, .z = 0.0},
            .m1 = {.x = 1.0, .y = 3.0, .z = 0.0},
        },
    };

    return autorigami::PiecewiseCubicHermiteSpline(segments);
}

void test_validation_accepts_known_valid_curves_and_rejects_known_invalid_curve() {
    constexpr double max_curvature = 2.5;
    constexpr double curvature_tolerance = 0.01;

    expect(
        autorigami::validate_curve_curvature(valid_curve_one(), max_curvature, curvature_tolerance),
        "first valid curve should pass curvature validation"
    );
    expect(
        autorigami::validate_curve_curvature(valid_curve_two(), max_curvature, curvature_tolerance),
        "second valid curve should pass curvature validation"
    );
    expect(
        !autorigami::validate_curve_curvature(invalid_curve(), max_curvature, curvature_tolerance),
        "invalid curve should fail curvature validation"
    );

    expect(
        autorigami::validate_piecewise_curve_curvature(
            valid_piecewise_curve(),
            max_curvature,
            curvature_tolerance
        ),
        "valid piecewise curve should pass curvature validation"
    );
    expect(
        !autorigami::validate_piecewise_curve_curvature(
            invalid_piecewise_curve(),
            max_curvature,
            curvature_tolerance
        ),
        "piecewise curve with one invalid segment should fail curvature validation"
    );
}

}  // namespace

int main() {
    try {
        test_validation_accepts_known_valid_curves_and_rejects_known_invalid_curve();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
