#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>

#include "distance_metrics.hpp"
#include "../optimizations/simd_operations.hpp"

template <typename Metric>
concept RawDistanceMetric = requires(const Metric& metric,
                                     std::span<const float> a,
                                     std::span<const float> b) {
    { metric.distance(a, b) } -> std::convertible_to<float>;
};

struct EuclideanMetricPolicy {
    float distance(std::span<const float> a, std::span<const float> b) const {
        return std::sqrt(simd_ops::squared_distance(a.data(), b.data(), a.size()));
    }
};

struct ManhattanMetricPolicy {
    float distance(std::span<const float> a, std::span<const float> b) const {
        return simd_ops::manhattan_distance(a.data(), b.data(), a.size());
    }
};

struct CosineMetricPolicy {
    float distance(std::span<const float> a, std::span<const float> b) const {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        const size_t dims = a.size();
        for (size_t i = 0; i < dims; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
        return 1.0f - dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
};

class VirtualMetricPolicy {
public:
    explicit VirtualMetricPolicy(std::shared_ptr<const DistanceMetric> metric)
        : metric_(std::move(metric)) {}

    float distance(std::span<const float> a, std::span<const float> b) const {
        return metric_->distance_raw(a, b);
    }

private:
    std::shared_ptr<const DistanceMetric> metric_;
};

static_assert(RawDistanceMetric<EuclideanMetricPolicy>);
static_assert(RawDistanceMetric<ManhattanMetricPolicy>);
static_assert(RawDistanceMetric<CosineMetricPolicy>);
static_assert(RawDistanceMetric<VirtualMetricPolicy>);
