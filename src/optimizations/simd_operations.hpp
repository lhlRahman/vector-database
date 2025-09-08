#pragma once

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include "../core/vector.hpp"

namespace simd_ops {
    float dot_product(const Vector& v1, const Vector& v2);

    void add(const Vector& v1, const Vector& v2, Vector& result);

    void subtract(const Vector& v1, const Vector& v2, Vector& result);
}
