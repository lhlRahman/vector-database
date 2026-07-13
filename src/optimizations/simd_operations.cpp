#include <cmath>     // std::abs(float) — do not rely on transitive includes
#include <stdexcept>

#include "simd_operations.hpp"

namespace simd_ops {

// ── Raw-pointer implementations (core SIMD kernels) ───────

#if defined(__ARM_NEON) || defined(__aarch64__)

float squared_distance(const float* a, const float* b, size_t size) {
    // 4 independent accumulators + fused multiply-add to hide FMA latency.
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f),
                s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        s0 = vfmaq_f32(s0, d0, d0);
        s1 = vfmaq_f32(s1, d1, d1);
        s2 = vfmaq_f32(s2, d2, d2);
        s3 = vfmaq_f32(s3, d3, d3);
    }
    for (; i + 4 <= size; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        s0 = vfmaq_f32(s0, d, d);
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    float result[4];
    vst1q_f32(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float dot_product(const float* a, const float* b, size_t size) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f),
                s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= size; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    float32x4_t sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    float result[4];
    vst1q_f32(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < size; ++i) total += a[i] * b[i];
    return total;
}

float manhattan_distance(const float* a, const float* b, size_t size) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        sum = vaddq_f32(sum, vabdq_f32(a_vec, b_vec));
    }
    float result[4];
    vst1q_f32(result, sum);
    float total = result[0] + result[1] + result[2] + result[3];
    for (; i < size; ++i) {
        total += std::abs(a[i] - b[i]);
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
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        vst1q_f32(r + i, vaddq_f32(a_vec, b_vec));
    }
    for (; i < size; ++i) r[i] = a[i] + b[i];
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
    for (; i + 4 <= size; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        vst1q_f32(r + i, vsubq_f32(a_vec, b_vec));
    }
    for (; i < size; ++i) r[i] = a[i] - b[i];
}

// Quantized uint8 L2: process 16 bytes at a time via NEON
uint32_t quantized_l2_u8(const uint8_t* a, const uint8_t* b, size_t size) {
    uint32x4_t sum = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t diff = vabdq_u8(va, vb);
        // Widen to 16-bit, multiply, accumulate to 32-bit
        uint16x8_t lo = vmull_u8(vget_low_u8(diff), vget_low_u8(diff));
        uint16x8_t hi = vmull_u8(vget_high_u8(diff), vget_high_u8(diff));
        sum = vaddq_u32(sum, vpaddlq_u16(lo));
        sum = vaddq_u32(sum, vpaddlq_u16(hi));
    }
    uint32_t result[4];
    vst1q_u32(result, sum);
    uint32_t total = result[0] + result[1] + result[2] + result[3];
    for (; i < size; ++i) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        total += static_cast<uint32_t>(d * d);
    }
    return total;
}

// Only take the AVX path when the compiler is actually targeting AVX. Without
// this guard the 256-bit intrinsics below are emitted on any x86 target and
// SIGILL on a CPU/VM without AVX (there is no runtime CPU-feature dispatch).
#elif (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && defined(__AVX__)

// Fused multiply-add when the target has FMA (build with -mfma), else mul+add.
static inline __m256 fma_ps(__m256 x, __m256 y, __m256 acc) {
#if defined(__FMA__)
    return _mm256_fmadd_ps(x, y, acc);  // x*y + acc, single rounding
#else
    return _mm256_add_ps(acc, _mm256_mul_ps(x, y));
#endif
}

float squared_distance(const float* a, const float* b, size_t size) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(),
           s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        s0 = fma_ps(d0, d0, s0);
        s1 = fma_ps(d1, d1, s1);
        s2 = fma_ps(d2, d2, s2);
        s3 = fma_ps(d3, d3, s3);
    }
    for (; i + 8 <= size; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        s0 = fma_ps(d, d, s0);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float result[8];
    _mm256_storeu_ps(result, s);
    float total = result[0] + result[1] + result[2] + result[3] +
                  result[4] + result[5] + result[6] + result[7];
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float dot_product(const float* a, const float* b, size_t size) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(),
           s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        s0 = fma_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      s0);
        s1 = fma_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  s1);
        s2 = fma_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = fma_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= size; i += 8)
        s0 = fma_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    float result[8];
    _mm256_storeu_ps(result, s);
    float total = result[0] + result[1] + result[2] + result[3] +
                  result[4] + result[5] + result[6] + result[7];
    for (; i < size; ++i) total += a[i] * b[i];
    return total;
}

float manhattan_distance(const float* a, const float* b, size_t size) {
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_add_ps(sum, _mm256_andnot_ps(sign_mask, diff));
    }
    float result[8];
    _mm256_storeu_ps(result, sum);
    float total = result[0] + result[1] + result[2] + result[3] +
                  result[4] + result[5] + result[6] + result[7];
    for (; i < size; ++i) {
        total += std::abs(a[i] - b[i]);
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
        _mm256_storeu_ps(r + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] + b[i];
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
        _mm256_storeu_ps(r + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] - b[i];
}

uint32_t quantized_l2_u8(const uint8_t* a, const uint8_t* b, size_t size) {
    // SSE2 fallback for x86 — process 16 bytes at a time
    uint32_t total = 0;
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        // Compute |a-b| using max(a,b) - min(a,b) for unsigned
        __m128i vmax = _mm_max_epu8(va, vb);
        __m128i vmin = _mm_min_epu8(va, vb);
        __m128i diff = _mm_sub_epi8(vmax, vmin);
        // Unpack to 16-bit and square
        __m128i lo = _mm_unpacklo_epi8(diff, _mm_setzero_si128());
        __m128i hi = _mm_unpackhi_epi8(diff, _mm_setzero_si128());
        __m128i sq_lo = _mm_madd_epi16(lo, lo);
        __m128i sq_hi = _mm_madd_epi16(hi, hi);
        __m128i sums = _mm_add_epi32(sq_lo, sq_hi);
        // Horizontal sum
        int32_t tmp[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), sums);
        total += static_cast<uint32_t>(tmp[0] + tmp[1] + tmp[2] + tmp[3]);
    }
    for (; i < size; ++i) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        total += static_cast<uint32_t>(d * d);
    }
    return total;
}

#else // scalar fallback

float squared_distance(const float* a, const float* b, size_t size) {
    float total = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float dot_product(const float* a, const float* b, size_t size) {
    float total = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

float manhattan_distance(const float* a, const float* b, size_t size) {
    float total = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        total += std::abs(a[i] - b[i]);
    }
    return total;
}

void add(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }
    for (size_t i = 0; i < v1.size(); ++i) result[i] = v1[i] + v2[i];
}

void subtract(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }
    for (size_t i = 0; i < v1.size(); ++i) result[i] = v1[i] - v2[i];
}

uint32_t quantized_l2_u8(const uint8_t* a, const uint8_t* b, size_t size) {
    uint32_t total = 0;
    for (size_t i = 0; i < size; ++i) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        total += static_cast<uint32_t>(d * d);
    }
    return total;
}

#endif

// ── Batched distance (prefetch-based 1-to-N) ─────────────

void batch_squared_distances(const float* query,
                             const float* const* candidates,
                             float* out_distances,
                             size_t num_candidates,
                             size_t dims) {
    for (size_t i = 0; i < num_candidates; ++i) {
        if (i + 1 < num_candidates) {
            __builtin_prefetch(candidates[i + 1], 0, 1);
        }
        out_distances[i] = squared_distance(query, candidates[i], dims);
    }
}

// ── Vector overloads (delegate to raw-pointer versions) ───

float squared_distance(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    return squared_distance(v1.data_ptr(), v2.data_ptr(), v1.size());
}

float dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    return dot_product(v1.data_ptr(), v2.data_ptr(), v1.size());
}

} // namespace simd_ops
