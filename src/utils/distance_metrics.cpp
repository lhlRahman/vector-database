#include "distance_metrics.hpp"
#include <cmath>
#include <stdexcept>

namespace DistanceMetrics {
    float euclidean(const Vector& v1, const Vector& v2){
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Vectors must be of the same size");
        }
        float sum = 0.0f;
        for (size_t i = 0; i < v1.size(); i++) {
            sum += pow(v1[i] - v2[i], 2);
        }
        return sqrt(sum);
    }

    float manhattan(const Vector& v1, const Vector& v2){
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Vectors must be of the same size");
        }
        float sum = 0.0f;
        for (size_t i = 0; i < v1.size(); i++) {
            sum += abs(v1[i] - v2[i]);
        }
        return sum;
    }

float cosine_similarity(const Vector& v1, const Vector& v2) {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Vectors must have the same dimension");
        }
        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            dot_product += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        return dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
    }

}