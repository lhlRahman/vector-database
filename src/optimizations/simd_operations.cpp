#include "simd_operations.hpp"

#include <atomic>
#include <cmath>
#include <stdexcept>

// Kernels are guarded by feature macros (not bare architecture macros) so the
// file builds at any baseline; dispatch falls back to scalar.
#if defined(__aarch64__) || defined(__ARM_NEON)
    #define VDB_HAVE_NEON 1
    #include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__SSE2__) || \
      (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define VDB_HAVE_SSE2 1
    #include <algorithm>
    #include <immintrin.h>
    #if defined(_MSC_VER)
        #include <intrin.h>
    #endif
    // AVX2 kernels are built per-function via target attribute (no global
    // -mavx2); MSVC compiles intrinsics unconditionally.
    #if defined(__GNUC__) || defined(__clang__)
        #define VDB_TARGET_AVX2 __attribute__((target("avx2,fma")))
        #define VDB_HAVE_AVX2_KERNELS 1
    #elif defined(_MSC_VER)
        #define VDB_TARGET_AVX2
        #define VDB_HAVE_AVX2_KERNELS 1
    #endif
#endif

namespace simd_ops {

namespace {

std::atomic<bool> g_simd_enabled{true};

// ── Scalar reference kernels (always compiled) ────────────

float squared_distance_scalar(const float* a, const float* b, std::size_t size) noexcept {
    float total = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float dot_product_scalar(const float* a, const float* b, std::size_t size) noexcept {
    float total = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

float manhattan_distance_scalar(const float* a, const float* b, std::size_t size) noexcept {
    float total = 0.0f;
    for (std::size_t i = 0; i < size; ++i) {
        total += std::abs(a[i] - b[i]);
    }
    return total;
}

void add_scalar(const float* a, const float* b, float* r, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) r[i] = a[i] + b[i];
}

void subtract_scalar(const float* a, const float* b, float* r, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) r[i] = a[i] - b[i];
}

std::uint64_t quantized_l2sq_u8_scalar(const std::uint8_t* a, const std::uint8_t* b,
                                       std::size_t size) noexcept {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < size; ++i) {
        std::int32_t d = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
        total += static_cast<std::uint64_t>(d * d);
    }
    return total;
}

#if defined(VDB_HAVE_NEON)

// ── NEON kernels (4 accumulators to hide FMA latency) ─────

inline float32x4_t fma_f32(float32x4_t acc, float32x4_t x, float32x4_t y) noexcept {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA)
    return vfmaq_f32(acc, x, y);
#else
    return vmlaq_f32(acc, x, y);
#endif
}

inline float hsum_f32(float32x4_t v) noexcept {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
#endif
}

float squared_distance_neon(const float* a, const float* b, std::size_t size) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i),      vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        acc0 = fma_f32(acc0, d0, d0);
        acc1 = fma_f32(acc1, d1, d1);
        acc2 = fma_f32(acc2, d2, d2);
        acc3 = fma_f32(acc3, d3, d3);
    }
    for (; i + 4 <= size; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = fma_f32(acc0, d, d);
    }
    float total = hsum_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float dot_product_neon(const float* a, const float* b, std::size_t size) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        acc0 = fma_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = fma_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = fma_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = fma_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= size; i += 4) {
        acc0 = fma_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float total = hsum_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

float manhattan_distance_neon(const float* a, const float* b, std::size_t size) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        acc0 = vaddq_f32(acc0, vabdq_f32(vld1q_f32(a + i),      vld1q_f32(b + i)));
        acc1 = vaddq_f32(acc1, vabdq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4)));
        acc2 = vaddq_f32(acc2, vabdq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8)));
        acc3 = vaddq_f32(acc3, vabdq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12)));
    }
    for (; i + 4 <= size; i += 4) {
        acc0 = vaddq_f32(acc0, vabdq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    float total = hsum_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < size; ++i) {
        total += std::abs(a[i] - b[i]);
    }
    return total;
}

void add_neon(const float* a, const float* b, float* r, std::size_t size) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        vst1q_f32(r + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] + b[i];
}

void subtract_neon(const float* a, const float* b, float* r, std::size_t size) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        vst1q_f32(r + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] - b[i];
}

std::uint64_t quantized_l2sq_u8_neon(const std::uint8_t* a, const std::uint8_t* b,
                                     std::size_t size) noexcept {
    // each u32 lane holds <= 4 * 255^2 per block, so folding to u64 every
    // block can't overflow
    uint64x2_t acc = vdupq_n_u64(0);
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        uint8x16_t diff = vabdq_u8(vld1q_u8(a + i), vld1q_u8(b + i));
        uint16x8_t lo = vmull_u8(vget_low_u8(diff), vget_low_u8(diff));
        uint16x8_t hi = vmull_u8(vget_high_u8(diff), vget_high_u8(diff));
        uint32x4_t block = vaddq_u32(vpaddlq_u16(lo), vpaddlq_u16(hi));
        acc = vpadalq_u32(acc, block);
    }
    std::uint64_t total = vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
    for (; i < size; ++i) {
        std::int32_t d = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
        total += static_cast<std::uint64_t>(d * d);
    }
    return total;
}

#endif // VDB_HAVE_NEON

#if defined(VDB_HAVE_SSE2)

// ── SSE2 kernels (x86 baseline) ───────────────────────────

inline float hsum_m128(__m128 v) noexcept {
    __m128 s = _mm_add_ps(v, _mm_movehl_ps(v, v));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

float squared_distance_sse2(const float* a, const float* b, std::size_t size) noexcept {
    __m128 acc0 = _mm_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m128 d0 = _mm_sub_ps(_mm_loadu_ps(a + i),      _mm_loadu_ps(b + i));
        __m128 d1 = _mm_sub_ps(_mm_loadu_ps(a + i + 4),  _mm_loadu_ps(b + i + 4));
        __m128 d2 = _mm_sub_ps(_mm_loadu_ps(a + i + 8),  _mm_loadu_ps(b + i + 8));
        __m128 d3 = _mm_sub_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(d0, d0));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(d1, d1));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(d2, d2));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(d3, d3));
    }
    for (; i + 4 <= size; i += 4) {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(d, d));
    }
    float total = hsum_m128(_mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3)));
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float dot_product_sse2(const float* a, const float* b, std::size_t size) noexcept {
    __m128 acc0 = _mm_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + i),      _mm_loadu_ps(b + i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a + i + 4),  _mm_loadu_ps(b + i + 4)));
        acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(a + i + 8),  _mm_loadu_ps(b + i + 8)));
        acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12)));
    }
    for (; i + 4 <= size; i += 4) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    float total = hsum_m128(_mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3)));
    for (; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

float manhattan_distance_sse2(const float* a, const float* b, std::size_t size) noexcept {
    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    __m128 acc0 = _mm_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m128 d0 = _mm_sub_ps(_mm_loadu_ps(a + i),      _mm_loadu_ps(b + i));
        __m128 d1 = _mm_sub_ps(_mm_loadu_ps(a + i + 4),  _mm_loadu_ps(b + i + 4));
        __m128 d2 = _mm_sub_ps(_mm_loadu_ps(a + i + 8),  _mm_loadu_ps(b + i + 8));
        __m128 d3 = _mm_sub_ps(_mm_loadu_ps(a + i + 12), _mm_loadu_ps(b + i + 12));
        acc0 = _mm_add_ps(acc0, _mm_andnot_ps(sign_mask, d0));
        acc1 = _mm_add_ps(acc1, _mm_andnot_ps(sign_mask, d1));
        acc2 = _mm_add_ps(acc2, _mm_andnot_ps(sign_mask, d2));
        acc3 = _mm_add_ps(acc3, _mm_andnot_ps(sign_mask, d3));
    }
    for (; i + 4 <= size; i += 4) {
        __m128 d = _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i));
        acc0 = _mm_add_ps(acc0, _mm_andnot_ps(sign_mask, d));
    }
    float total = hsum_m128(_mm_add_ps(_mm_add_ps(acc0, acc1), _mm_add_ps(acc2, acc3)));
    for (; i < size; ++i) {
        total += std::abs(a[i] - b[i]);
    }
    return total;
}

void add_sse2(const float* a, const float* b, float* r, std::size_t size) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        _mm_storeu_ps(r + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] + b[i];
}

void subtract_sse2(const float* a, const float* b, float* r, std::size_t size) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        _mm_storeu_ps(r + i, _mm_sub_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] - b[i];
}

std::uint64_t quantized_l2sq_u8_sse2(const std::uint8_t* a, const std::uint8_t* b,
                                     std::size_t size) noexcept {
    const __m128i zero = _mm_setzero_si128();
    __m128i acc64 = zero;
    std::size_t i = 0;
    while (i + 16 <= size) {
        // 4096 blocks * 4 * 255^2 per u32 lane stays under 2^31, then fold to u64
        std::size_t chunk_end = std::min(size, i + 16 * 4096);
        __m128i acc32 = zero;
        for (; i + 16 <= chunk_end; i += 16) {
            __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
            __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
            __m128i diff = _mm_sub_epi8(_mm_max_epu8(va, vb), _mm_min_epu8(va, vb));
            __m128i lo = _mm_unpacklo_epi8(diff, zero);
            __m128i hi = _mm_unpackhi_epi8(diff, zero);
            acc32 = _mm_add_epi32(acc32, _mm_add_epi32(_mm_madd_epi16(lo, lo),
                                                       _mm_madd_epi16(hi, hi)));
        }
        acc64 = _mm_add_epi64(acc64, _mm_unpacklo_epi32(acc32, zero));
        acc64 = _mm_add_epi64(acc64, _mm_unpackhi_epi32(acc32, zero));
    }
    alignas(16) std::uint64_t lanes[2];
    _mm_store_si128(reinterpret_cast<__m128i*>(lanes), acc64);
    std::uint64_t total = lanes[0] + lanes[1];
    for (; i < size; ++i) {
        std::int32_t d = static_cast<std::int32_t>(a[i]) - static_cast<std::int32_t>(b[i]);
        total += static_cast<std::uint64_t>(d * d);
    }
    return total;
}

#endif // VDB_HAVE_SSE2

#if defined(VDB_HAVE_AVX2_KERNELS)

// ── AVX2+FMA kernels (selected at runtime) ────────────────

VDB_TARGET_AVX2 inline float hsum_m256(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

VDB_TARGET_AVX2 float squared_distance_avx2(const float* a, const float* b,
                                            std::size_t size) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    for (; i + 8 <= size; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }
    float total = hsum_m256(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                          _mm256_add_ps(acc2, acc3)));
    for (; i < size; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

VDB_TARGET_AVX2 float dot_product_avx2(const float* a, const float* b,
                                       std::size_t size) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= size; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float total = hsum_m256(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                          _mm256_add_ps(acc2, acc3)));
    for (; i < size; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

VDB_TARGET_AVX2 float manhattan_distance_avx2(const float* a, const float* b,
                                              std::size_t size) noexcept {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    std::size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, d0));
        acc1 = _mm256_add_ps(acc1, _mm256_andnot_ps(sign_mask, d1));
        acc2 = _mm256_add_ps(acc2, _mm256_andnot_ps(sign_mask, d2));
        acc3 = _mm256_add_ps(acc3, _mm256_andnot_ps(sign_mask, d3));
    }
    for (; i + 8 <= size; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_add_ps(acc0, _mm256_andnot_ps(sign_mask, d));
    }
    float total = hsum_m256(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                          _mm256_add_ps(acc2, acc3)));
    for (; i < size; ++i) {
        total += std::abs(a[i] - b[i]);
    }
    return total;
}

VDB_TARGET_AVX2 void add_avx2(const float* a, const float* b, float* r,
                              std::size_t size) noexcept {
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        _mm256_storeu_ps(r + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] + b[i];
}

VDB_TARGET_AVX2 void subtract_avx2(const float* a, const float* b, float* r,
                                   std::size_t size) noexcept {
    std::size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        _mm256_storeu_ps(r + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < size; ++i) r[i] = a[i] - b[i];
}

bool cpu_has_avx2_fma() noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    const bool avx     = (r[2] & (1 << 28)) != 0;
    const bool fma     = (r[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) return false;
    // OS must save XMM+YMM state
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

#endif // VDB_HAVE_AVX2_KERNELS

// ── Runtime dispatch ──────────────────────────────────────

struct Kernels {
    float (*squared_distance)(const float*, const float*, std::size_t) noexcept;
    float (*dot_product)(const float*, const float*, std::size_t) noexcept;
    float (*manhattan_distance)(const float*, const float*, std::size_t) noexcept;
    void (*add)(const float*, const float*, float*, std::size_t) noexcept;
    void (*subtract)(const float*, const float*, float*, std::size_t) noexcept;
    std::uint64_t (*quantized_l2sq_u8)(const std::uint8_t*, const std::uint8_t*,
                                       std::size_t) noexcept;
};

Kernels select_kernels() noexcept {
#if defined(VDB_HAVE_NEON)
    return {squared_distance_neon, dot_product_neon, manhattan_distance_neon,
            add_neon, subtract_neon, quantized_l2sq_u8_neon};
#elif defined(VDB_HAVE_SSE2)
    #if defined(VDB_HAVE_AVX2_KERNELS)
    if (cpu_has_avx2_fma()) {
        return {squared_distance_avx2, dot_product_avx2, manhattan_distance_avx2,
                add_avx2, subtract_avx2, quantized_l2sq_u8_sse2};
    }
    #endif
    return {squared_distance_sse2, dot_product_sse2, manhattan_distance_sse2,
            add_sse2, subtract_sse2, quantized_l2sq_u8_sse2};
#else
    return {squared_distance_scalar, dot_product_scalar, manhattan_distance_scalar,
            add_scalar, subtract_scalar, quantized_l2sq_u8_scalar};
#endif
}

const Kernels& kernels() noexcept {
    static const Kernels k = select_kernels();
    return k;
}

void check_same_size(const Vector& v1, const Vector& v2, const Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }
}

} // namespace

// ── Public API ────────────────────────────────────────────

void set_enabled(bool enabled) noexcept {
    g_simd_enabled.store(enabled, std::memory_order_relaxed);
}

bool is_enabled() noexcept {
    return g_simd_enabled.load(std::memory_order_relaxed);
}

float squared_distance(const float* a, const float* b, std::size_t size) noexcept {
    if (!is_enabled()) return squared_distance_scalar(a, b, size);
    return kernels().squared_distance(a, b, size);
}

float dot_product(const float* a, const float* b, std::size_t size) noexcept {
    if (!is_enabled()) return dot_product_scalar(a, b, size);
    return kernels().dot_product(a, b, size);
}

float manhattan_distance(const float* a, const float* b, std::size_t size) noexcept {
    if (!is_enabled()) return manhattan_distance_scalar(a, b, size);
    return kernels().manhattan_distance(a, b, size);
}

std::uint64_t quantized_l2sq_u8(const std::uint8_t* a, const std::uint8_t* b,
                                std::size_t size) noexcept {
    if (!is_enabled()) return quantized_l2sq_u8_scalar(a, b, size);
    return kernels().quantized_l2sq_u8(a, b, size);
}

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

void add(const Vector& v1, const Vector& v2, Vector& result) {
    check_same_size(v1, v2, result);
    if (!is_enabled()) {
        add_scalar(v1.data_ptr(), v2.data_ptr(), result.data_ptr(), v1.size());
    } else {
        kernels().add(v1.data_ptr(), v2.data_ptr(), result.data_ptr(), v1.size());
    }
}

void subtract(const Vector& v1, const Vector& v2, Vector& result) {
    check_same_size(v1, v2, result);
    if (!is_enabled()) {
        subtract_scalar(v1.data_ptr(), v2.data_ptr(), result.data_ptr(), v1.size());
    } else {
        kernels().subtract(v1.data_ptr(), v2.data_ptr(), result.data_ptr(), v1.size());
    }
}

} // namespace simd_ops
