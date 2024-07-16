// src/optimizations/simd_operations.hpp

#pragma once
#include "../core/vector.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace simd_ops {
    float dot_product(const Vector& v1, const Vector& v2);

    void add(const Vector& v1, const Vector& v2, Vector& result);

    void subtract(const Vector& v1, const Vector& v2, Vector& result);
}
