#pragma once

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

#include <cstddef>
#include <cstdint>
#include "../core/vector.hpp"

namespace simd_ops {

// ── Single-pair operations (Vector overloads) ─────────────
float dot_product(const Vector& v1, const Vector& v2);
float squared_distance(const Vector& v1, const Vector& v2);
void add(const Vector& v1, const Vector& v2, Vector& result);
void subtract(const Vector& v1, const Vector& v2, Vector& result);

// ── Single-pair operations (raw-pointer, zero-copy) ───────
float squared_distance(const float* a, const float* b, size_t size);
float dot_product(const float* a, const float* b, size_t size);
float manhattan_distance(const float* a, const float* b, size_t size);

// ── Batched distance (1-to-N with prefetch) ───────────────
// Computes squared Euclidean distance from query to each candidate.
// Prefetches next candidate while computing current one.
void batch_squared_distances(const float* query,
                             const float* const* candidates,
                             float* out_distances,
                             size_t num_candidates,
                             size_t dims);

// ── Scalar-quantized distance (uint8) ─────────────────────
// Approximate L2 distance between quantized vectors.
// Returns sum of squared differences of uint8 values.
uint32_t quantized_l2_u8(const uint8_t* a, const uint8_t* b, size_t size);

} // namespace simd_ops
