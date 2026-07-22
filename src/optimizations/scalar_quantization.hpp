#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "simd_operations.hpp"

// Scalar quantization: float32 → uint8. Per-dimension offset (trained min),
// one shared scale = 255 / widest range. Offsets cancel in differences, so
// quantized squared L2 ≈ scale^2 * true squared L2 and ranking is preserved.
// (Per-dim scales would reweight each dimension by 1/range^2 and distort the
// metric.) Use for candidate filtering, re-rank the top hits with exact
// float distance.
class ScalarQuantizer {
public:
    ScalarQuantizer() = default;

    explicit ScalarQuantizer(size_t dims)
        : dims_(dims), mins_(dims, std::numeric_limits<float>::max()),
          maxs_(dims, std::numeric_limits<float>::lowest()) {}

    // Train: compute per-dimension min/max and the shared scale.
    void train(const float* const* vectors, size_t count) {
        if (count == 0) return;

        std::fill(mins_.begin(), mins_.end(), std::numeric_limits<float>::max());
        std::fill(maxs_.begin(), maxs_.end(), std::numeric_limits<float>::lowest());

        for (size_t i = 0; i < count; ++i) {
            const float* v = vectors[i];
            for (size_t d = 0; d < dims_; ++d) {
                mins_[d] = std::min(mins_[d], v[d]);
                maxs_[d] = std::max(maxs_[d], v[d]);
            }
        }

        float max_range = 0.0f;
        for (size_t d = 0; d < dims_; ++d) {
            max_range = std::max(max_range, maxs_[d] - mins_[d]);
        }
        scale_ = (max_range > 0.0f) ? 255.0f / max_range : 0.0f;

        trained_ = true;
    }

    // Quantize a single float vector into the provided uint8 buffer.
    void quantize(const float* src, uint8_t* dst) const {
        for (size_t d = 0; d < dims_; ++d) {
            float val = (src[d] - mins_[d]) * scale_;
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

    // Squared L2 in code space (≈ scale^2 * true squared L2).
    uint64_t distance_quantized(const uint8_t* a, const uint8_t* b) const {
        return simd_ops::quantized_l2sq_u8(a, b, dims_);
    }

    // Approximate float squared L2 — for ranking, not exact values.
    float approximate_distance_sq(uint64_t quant_dist) const {
        if (scale_ <= 0.0f) return 0.0f;
        return static_cast<float>(quant_dist) / (scale_ * scale_);
    }

    bool is_trained() const { return trained_; }
    size_t dims() const { return dims_; }
    size_t quantized_size() const { return dims_; }

private:
    size_t dims_ = 0;
    std::vector<float> mins_;
    std::vector<float> maxs_;
    float scale_ = 0.0f;
    bool trained_ = false;
};
