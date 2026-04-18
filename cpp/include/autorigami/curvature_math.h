#pragma once

#include "autorigami/splines.h"

#include <boost/math/tools/roots.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace autorigami {

double solve_root_toms748(
    const CubicPowerBasisSegment& seg,
    double left,
    double right);

}  // namespace autorigami
