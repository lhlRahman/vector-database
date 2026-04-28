#include <cmath>
#include <stdexcept>

#include "distance_metrics.hpp"
#include "../optimizations/simd_operations.hpp"

// ── Euclidean ─────────────────────────────────────────────

float EuclideanDistance::distance(const Vector& v1, const Vector& v2) const {
    return std::sqrt(simd_ops::squared_distance(v1, v2));
}

float EuclideanDistance::distance_raw(std::span<const float> a, std::span<const float> b) const {
    return std::sqrt(simd_ops::squared_distance(a.data(), b.data(), a.size()));
}

// ── Manhattan ─────────────────────────────────────────────

float ManhattanDistance::distance(const Vector& v1, const Vector& v2) const {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    return distance_raw(std::span(v1.data_ptr(), v1.size()),
                        std::span(v2.data_ptr(), v2.size()));
}

float ManhattanDistance::distance_raw(std::span<const float> a, std::span<const float> b) const {
    return simd_ops::manhattan_distance(a.data(), b.data(), a.size());
}

// ── Cosine ────────────────────────────────────────────────

float CosineSimilarity::distance(const Vector& v1, const Vector& v2) const {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    return distance_raw(std::span(v1.data_ptr(), v1.size()),
                        std::span(v2.data_ptr(), v2.size()));
}

float CosineSimilarity::distance_raw(std::span<const float> a, std::span<const float> b) const {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    const size_t dims = a.size();
    for (size_t i = 0; i < dims; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 1.0f;
    return 1.0f - dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}
