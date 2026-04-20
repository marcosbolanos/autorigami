#include "autorigami/generator.h"

#include <utility>

namespace autorigami {

PiecewiseHermiteData piecewise_hermite_generator() {
    return PiecewiseHermiteData{
        .points =
            {
                {.x = 0.0, .y = 0.0, .z = 0.0},
                {.x = 1.0, .y = 0.4, .z = -0.2},
                {.x = 2.1, .y = -0.3, .z = 0.6},
                {.x = 3.0, .y = 0.1, .z = 1.0},
            },
        .tangents =
            {
                {.x = 0.8, .y = 0.2, .z = -0.1},
                {.x = 0.9, .y = -0.5, .z = 0.4},
                {.x = 0.7, .y = 0.6, .z = 0.3},
                {.x = 0.5, .y = 0.2, .z = 0.4},
            },
    };
}

}  // namespace autorigami
