#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "autorigami/validation/nonlocal_distance.h"

namespace {

using autorigami::NonlocalDistanceValidationResult;
using autorigami::Vec3;

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_no_violations_for_well_spaced_segments() {
    const std::vector<Vec3> points = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {3.0, 2.0, 0.0},
    };
    const NonlocalDistanceValidationResult result =
        autorigami::validate_polyline_nonlocal_distance(points, 0.5, 0.2, false);
    expect(result.violation_count == 0, "expected no violations");
    expect(std::isfinite(result.minimum_checked_distance), "expected finite min distance");
    expect(result.minimum_checked_distance >= 0.5, "expected min distance >= threshold");
}

void test_detects_violation_when_nonlocal_segments_are_close() {
    const std::vector<Vec3> points = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {0.2, 0.2, 0.0},
    };
    const NonlocalDistanceValidationResult result =
        autorigami::validate_polyline_nonlocal_distance(points, 0.5, 0.1, false);
    expect(result.violation_count > 0, "expected at least one violation");
    expect(std::isfinite(result.minimum_checked_distance), "expected finite min distance");
    expect(result.minimum_checked_distance < 0.5, "expected min distance below threshold");
}

void test_stop_on_first_violation() {
    const std::vector<Vec3> points = {
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},
        {0.2, 0.2, 0.0},
        {2.0, 0.2, 0.0},
    };
    const NonlocalDistanceValidationResult early =
        autorigami::validate_polyline_nonlocal_distance(points, 0.5, 0.1, true);
    const NonlocalDistanceValidationResult full =
        autorigami::validate_polyline_nonlocal_distance(points, 0.5, 0.1, false);
    expect(early.violation_count == 1, "expected early-exit single violation");
    expect(full.violation_count >= early.violation_count, "expected full count >= early count");
}

void test_invalid_arguments_throw() {
    bool threw = false;
    try {
        const std::vector<Vec3> points = {{0.0, 0.0, 0.0}};
        static_cast<void>(autorigami::validate_polyline_nonlocal_distance(points, 0.5, 0.1, false));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "expected invalid_argument for short input");
}

}  // namespace

int main() {
    test_no_violations_for_well_spaced_segments();
    test_detects_violation_when_nonlocal_segments_are_close();
    test_stop_on_first_violation();
    test_invalid_arguments_throw();
    return 0;
}
