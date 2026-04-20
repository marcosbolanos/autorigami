#include "autorigami/generator/energy.h"

namespace autorigami {

double evaluate_end_of_spline_energy(
    const PiecewiseHermiteData& spline,
    const GeneratorAxis& axis
) {
    (void)spline;
    (void)axis;
    // Placeholder objective. Real energy terms get added here.
    return 0.0;
}

}  // namespace autorigami
