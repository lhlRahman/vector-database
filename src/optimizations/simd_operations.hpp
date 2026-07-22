#pragma once

#include <cstddef>
#include <cstdint>

#include "../core/vector.hpp"

// SIMD vector kernels. NEON on ARM; SSE2 baseline on x86 with AVX2+FMA
// picked at runtime, so no -mavx* build flags are needed. Float results can
// differ slightly from scalar (reassociation/FMA); integer kernels are exact.
namespace simd_ops {

// ── Runtime toggle (scalar path when disabled) ────────────
void set_enabled(bool enabled) noexcept;
bool is_enabled() noexcept;

// ── Single-pair operations (Vector overloads) ─────────────
float dot_product(const Vector& v1, const Vector& v2);
float squared_distance(const Vector& v1, const Vector& v2);
void add(const Vector& v1, const Vector& v2, Vector& result);
void subtract(const Vector& v1, const Vector& v2, Vector& result);

// ── Single-pair operations (raw-pointer, zero-copy) ───────
float squared_distance(const float* a, const float* b, std::size_t size) noexcept;
float dot_product(const float* a, const float* b, std::size_t size) noexcept;
float manhattan_distance(const float* a, const float* b, std::size_t size) noexcept;

// ── Scalar-quantized distance (uint8) ─────────────────────
// Squared L2 in code space, no sqrt. u64 accumulator can't overflow.
std::uint64_t quantized_l2sq_u8(const std::uint8_t* a, const std::uint8_t* b,
                                std::size_t size) noexcept;

} // namespace simd_ops
