#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "simd_operations.hpp"

// Scalar quantization: float32 → uint8 with a SINGLE global scale.
//
//   q(x) = (x - global_min) * (255 / (global_max - global_min))
//
// A global (uniform) scale is deliberate: the quantized squared L2 is then
//   sum_d (q(a_d) - q(b_d))^2 = scale^2 * sum_d (a_d - b_d)^2 = scale^2 * ||a-b||^2,
// i.e. PROPORTIONAL to the true squared L2. A per-dimension scale (the previous
// design) weights each dim by (255/range_d)^2, turning it into a *distorted* L2
// that silently costs recall. Use for fast candidate filtering, then re-rank the
// top results with exact float distance.
class ScalarQuantizer {
public:
    ScalarQuantizer() = default;

    explicit ScalarQuantizer(size_t dims) : dims_(dims) {}

    // Train: compute a single global [min, max] across ALL dimensions of all
    // training vectors.
    void train(const float* const* vectors, size_t count) {
        if (count == 0) return;
        float lo = std::numeric_limits<float>::max();
        float hi = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < count; ++i) {
            const float* v = vectors[i];
            for (size_t d = 0; d < dims_; ++d) {
                lo = std::min(lo, v[d]);
                hi = std::max(hi, v[d]);
            }
        }
        min_ = lo;
        max_ = hi;
        const float range = max_ - min_;
        scale_ = (range > 0.0f) ? 255.0f / range : 0.0f;
        trained_ = true;
    }

    // Quantize a single float vector into the provided uint8 buffer.
    void quantize(const float* src, uint8_t* dst) const {
        for (size_t d = 0; d < dims_; ++d) {
            float val = (src[d] - min_) * scale_;
            val = std::max(0.0f, std::min(255.0f, val));
            dst[d] = static_cast<uint8_t>(val + 0.5f);
        }
    }

    // Quantize N vectors into a contiguous buffer (N * dims_ bytes).
    void quantize_batch(const float* const* vectors, uint8_t* dst, size_t count) const {
        for (size_t i = 0; i < count; ++i) {
            quantize(vectors[i], dst + i * dims_);
        }
    }

    // Approximate squared-L2 between two quantized vectors via integer SIMD.
    // With the global scale this is proportional to the true squared distance.
    uint32_t distance_quantized(const uint8_t* a, const uint8_t* b) const {
        return simd_ops::quantized_l2_u8(a, b, dims_);
    }

    // Convert a quantized squared distance back to an approximate float squared
    // distance. Exact inverse of the uniform scale (up to rounding); ranking-only.
    float approximate_distance_sq(uint32_t quant_dist) const {
        if (scale_ <= 0.0f) return 0.0f;
        return static_cast<float>(quant_dist) / (scale_ * scale_);
    }

    bool is_trained() const { return trained_; }
    size_t dims() const { return dims_; }
    size_t quantized_size() const { return dims_; }

private:
    size_t dims_ = 0;
    float min_ = 0.0f;
    float max_ = 0.0f;
    float scale_ = 0.0f;  // single global scale (uniform across dimensions)
    bool trained_ = false;
};
