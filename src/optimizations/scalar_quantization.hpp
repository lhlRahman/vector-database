#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "simd_operations.hpp"

// Scalar quantization: float32 → uint8 per dimension.
// Each dimension gets its own [min, max] range.
// Quantized value = (float - min) / (max - min) * 255.
//
// Distance between quantized vectors approximates true L2.
// Use for fast candidate filtering, then re-rank top results with exact float distance.
class ScalarQuantizer {
public:
    ScalarQuantizer() = default;

    explicit ScalarQuantizer(size_t dims)
        : dims_(dims), mins_(dims, std::numeric_limits<float>::max()),
          maxs_(dims, std::numeric_limits<float>::lowest()),
          scales_(dims, 0.0f), trained_(false) {}

    // Train: compute per-dimension min/max from a set of vectors (raw pointers).
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

        for (size_t d = 0; d < dims_; ++d) {
            float range = maxs_[d] - mins_[d];
            scales_[d] = (range > 0.0f) ? 255.0f / range : 0.0f;
        }

        trained_ = true;
    }

    // Quantize a single float vector into the provided uint8 buffer.
    void quantize(const float* src, uint8_t* dst) const {
        for (size_t d = 0; d < dims_; ++d) {
            float val = (src[d] - mins_[d]) * scales_[d];
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

    // Approximate L2² distance between two quantized vectors using integer SIMD.
    // Returns a value proportional to the true squared distance.
    uint32_t distance_quantized(const uint8_t* a, const uint8_t* b) const {
        return simd_ops::quantized_l2_u8(a, b, dims_);
    }

    // Convert quantized distance back to approximate float distance (squared).
    // This is approximate — use only for ranking, not exact values.
    float approximate_distance_sq(uint32_t quant_dist) const {
        // Average scale factor across dimensions
        float avg_range_sq = 0.0f;
        for (size_t d = 0; d < dims_; ++d) {
            float range = maxs_[d] - mins_[d];
            avg_range_sq += range * range;
        }
        avg_range_sq /= static_cast<float>(dims_);
        return static_cast<float>(quant_dist) * avg_range_sq / (255.0f * 255.0f);
    }

    bool is_trained() const { return trained_; }
    size_t dims() const { return dims_; }
    size_t quantized_size() const { return dims_; }

private:
    size_t dims_ = 0;
    std::vector<float> mins_;
    std::vector<float> maxs_;
    std::vector<float> scales_;
    bool trained_ = false;
};
