#include "../version.h"
#if !defined(BACKPROP_TOOLS_UTILS_GENERIC_MATH_H) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#define BACKPROP_TOOLS_UTILS_GENERIC_MATH_H

#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::math {

    template<typename T>
    constexpr T PI = 3.141592653589793238462643383279502884L;

    template<typename T>
    constexpr T FRAC_2_SQRTPI = 1.128379167095512573896158903121545172L;
    template<typename T>
    constexpr T SQRT1_2 = 0.707106781186547524400844362104849039L;
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
