#include <stdexcept>

#include "simd_operations.hpp"

namespace simd_ops {

#if defined(__ARM_NEON) || defined(__aarch64__)
namespace {
float scalar_dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    float result = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}
} // namespace
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)

// Runtime check for NEON support
bool has_neon_support() {
    // On Apple Silicon (M1/M2), NEON is always available
    #if defined(__aarch64__) && defined(__APPLE__)
        return true;
    #else
        // For other ARM systems, we'd need to check CPU features
        // This is a simplified check - in production you might want more robust detection
        return true;
    #endif
}

float dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    
    if (!has_neon_support()) {
        // Fallback to scalar implementation
        return scalar_dot_product(v1, v2);
    }
    
    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    size_t size = v1.size();

    float32x4_t sum = vdupq_n_f32(0.0f);

    // Process 4 floats at a time using NEON
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        sum = vaddq_f32(sum, vmulq_f32(a_vec, b_vec));
    }

    // Horizontal sum of the 4 float values
    float result[4];
    vst1q_f32(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

float squared_distance(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    size_t size = v1.size();

    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(a_vec, b_vec);
        sum = vaddq_f32(sum, vmulq_f32(diff, diff));
    }

    float result[4];
    vst1q_f32(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

void add(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    if (!has_neon_support()) {
        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] + v2[i];
        }
        return;
    }

    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    float* r = result.data_ptr();
    size_t size = v1.size();

    // Process 4 floats at a time using NEON
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t r_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(r + i, r_vec);
    }
    for (; i < size; ++i) {
        r[i] = a[i] + b[i];
    }
}

void subtract(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    if (!has_neon_support()) {
        // Fallback to scalar implementation
        for (size_t i = 0; i < v1.size(); ++i) {
            result[i] = v1[i] - v2[i];
        }
        return;
    }

    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    float* r = result.data_ptr();
    size_t size = v1.size();

    // Process 4 floats at a time using NEON
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t r_vec = vsubq_f32(a_vec, b_vec);
        vst1q_f32(r + i, r_vec);
    }
    for (; i < size; ++i) {
        r[i] = a[i] - b[i];
    }
}

#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

float squared_distance(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    size_t size = v1.size();

    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    float total = result[0] + result[1] + result[2] + result[3] +
                  result[4] + result[5] + result[6] + result[7];
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    size_t size = v1.size();

    __m256 sum = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    float total = result[0] + result[1] + result[2] + result[3] +
           result[4] + result[5] + result[6] + result[7];
    for (; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

void add(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }

    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    float* r = result.data_ptr();
    size_t size = v1.size();

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 r_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(r + i, r_vec);
    }
    for (; i < size; ++i) {
        r[i] = a[i] + b[i];
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

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 r_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(r + i, r_vec);
    }
    for (; i < size; ++i) {
        r[i] = a[i] - b[i];
    }
}

#else

float squared_distance(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    float total = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        total += diff * diff;
    }
    return total;
}

float dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    float result = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
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

} // namespace simd_ops
