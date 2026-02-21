#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "distance_metrics.hpp"
#include "../optimizations/simd_operations.hpp"

float EuclideanDistance::distance(const Vector& v1, const Vector& v2) const {
    // SIMD-accelerated single-pass squared distance
    return std::sqrt(simd_ops::squared_distance(v1, v2));
}

float ManhattanDistance::distance(const Vector& v1, const Vector& v2) const {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same dimension");
    }
    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum += std::abs(v1[i] - v2[i]);
    }
    return sum;
}

float CosineSimilarity::distance(const Vector& v1, const Vector& v2) const {
    // Single-pass: compute dot, norm1_sq, norm2_sq simultaneously
    float dot = 0.0f, norm1_sq = 0.0f, norm2_sq = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float a = v1[i], b = v2[i];
        dot += a * b;
        norm1_sq += a * a;
        norm2_sq += b * b;
    }
    if (norm1_sq == 0.0f || norm2_sq == 0.0f) {
        return 1.0f;
    }
    return 1.0f - dot / (std::sqrt(norm1_sq) * std::sqrt(norm2_sq));
}
