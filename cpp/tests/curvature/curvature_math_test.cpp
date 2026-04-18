#include "autorigami/curvature_math.h"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

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

void expect_contains(
    double lower,
    double upper,
    double value,
    double tolerance,
    const char* message
) {
    if (value < lower - tolerance || value > upper + tolerance) {
        throw std::runtime_error(message);
    }
}

void test_interval_solver_bounds_quarter_parameter_maximum() {
    const autorigami::CubicPowerBasisSegment segment{
        .a = {.x = 0.0, .y = 0.0, .z = 0.0},
        .b = {.x = 0.0, .y = 1.0, .z = 0.0},
        .c = {.x = 1.0, .y = -0.5, .z = 0.0},
        .d = {.x = 0.0, .y = 0.0625, .z = 0.0},
    };

    const autorigami::CurvatureMaxResult result = autorigami::max_curvature_of_segment(segment);

    expect_contains(
        result.lower_bound,
        result.upper_bound,
        2.0,
        1e-8,
        "expected certified bounds to contain curvature 2"
    );
    expect(result.upper_bound >= result.lower_bound, "expected upper bound to exceed lower bound");
    expect(
        result.upper_bound - result.lower_bound <= 1e-5,
        "expected certified bounds to be tight for the quarter-shifted parabola"
    );
    expect_near(result.best_t, 0.25, 1e-3, "expected best sampled t near the interior maximum");
}

void test_interval_solver_bounds_three_quarter_parameter_maximum() {
    const autorigami::CubicPowerBasisSegment segment{
        .a = {.x = 0.0, .y = 0.0, .z = 0.0},
        .b = {.x = 0.0, .y = 1.0, .z = 0.0},
        .c = {.x = 1.0, .y = -1.5, .z = 0.0},
        .d = {.x = 0.0, .y = 0.5625, .z = 0.0},
    };

    const autorigami::CurvatureMaxResult result = autorigami::max_curvature_of_segment(segment);

    expect_contains(
        result.lower_bound,
        result.upper_bound,
        2.0,
        1e-8,
        "expected certified bounds to contain curvature 2"
    );
    expect(
        result.upper_bound - result.lower_bound <= 1e-5,
        "expected certified bounds to be tight for the three-quarter-shifted parabola"
    );
    expect_near(result.best_t, 0.75, 1e-3, "expected best sampled t near the interior maximum");
}

void test_interval_solver_bounds_endpoint_maximum() {
    const autorigami::CubicPowerBasisSegment segment{
        .a = {.x = 0.0, .y = 0.0, .z = 0.0},
        .b = {.x = 0.0, .y = 1.0, .z = 0.0},
        .c = {.x = 1.0, .y = 0.0, .z = 0.0},
        .d = {.x = 0.0, .y = 0.0, .z = 0.0},
    };

    const autorigami::CurvatureMaxResult result = autorigami::max_curvature_of_segment(segment);

    expect_contains(
        result.lower_bound,
        result.upper_bound,
        2.0,
        1e-8,
        "expected certified bounds to contain curvature 2"
    );
    expect(
        result.upper_bound - result.lower_bound <= 1e-5,
        "expected certified bounds to be tight for the endpoint parabola"
    );
    expect_near(result.best_t, 0.0, 1e-9, "expected endpoint maximum to be sampled at t = 0");
}

}  // namespace

int main() {
    try {
        test_interval_solver_bounds_quarter_parameter_maximum();
        test_interval_solver_bounds_three_quarter_parameter_maximum();
        test_interval_solver_bounds_endpoint_maximum();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
