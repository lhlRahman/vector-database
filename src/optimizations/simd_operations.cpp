// src/optimizations/simd_operations.cpp

#include "simd_operations.hpp"
#include <stdexcept>

namespace simd_ops {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
float dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    size_t size = v1.size();

    __m256 sum = _mm256_setzero_ps();

    for (size_t i = 0; i < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    return result[0] + result[1] + result[2] + result[3] +
           result[4] + result[5] + result[6] + result[7];
}

void add(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    float* r = result.data_ptr();
    size_t size = v1.size();

    for (size_t i = 0; i < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 r_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(r + i, r_vec);
    }
}

void subtract(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    float* r = result.data_ptr();
    size_t size = v1.size();

    for (size_t i = 0; i < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 r_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(r + i, r_vec);
    }
}

#else

float dot_product(const Vector& v1, const Vector& v2) {
    return Vector::dot_product(v1, v2);
}

void add(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
}

void subtract(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
}

#endif

}
