#pragma once

#include "autorigami/generator/optimize_end_of_spline.h"

namespace autorigami {

[[nodiscard]] double evaluate_end_of_spline_energy(
    const PiecewiseHermiteData& spline,
    const GeneratorAxis& axis
);

}  // namespace autorigami
