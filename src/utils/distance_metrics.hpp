#pragma once

#include <cstddef>
#include <span>
#include "../core/vector.hpp"

class DistanceMetric {
public:
    virtual float distance(const Vector& v1, const Vector& v2) const = 0;

    // Span overload — avoids creating Vector objects (zero-copy from mmap).
    // Spans must be the same length; behaviour is undefined otherwise.
    virtual float distance_raw(std::span<const float> a, std::span<const float> b) const = 0;

    virtual ~DistanceMetric() = default;
};

class EuclideanDistance : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
    float distance_raw(std::span<const float> a, std::span<const float> b) const override;
};

class ManhattanDistance : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
    float distance_raw(std::span<const float> a, std::span<const float> b) const override;
};

class CosineSimilarity : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
    float distance_raw(std::span<const float> a, std::span<const float> b) const override;
};
