#include "distance_metrics.hpp"
#include <cmath>
#include <stdexcept>

float EuclideanDistance::distance(const Vector& v1, const Vector& v2) const {
    return std::sqrt(Vector::simd_dot_product(v1, v1) + 
                     Vector::simd_dot_product(v2, v2) - 
                     2 * Vector::simd_dot_product(v1, v2));
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
    float dot_product = Vector::simd_dot_product(v1, v2);
    float norm1 = std::sqrt(Vector::simd_dot_product(v1, v1));
    float norm2 = std::sqrt(Vector::simd_dot_product(v2, v2));
    return 1.0f - dot_product / (norm1 * norm2);
}