#pragma once

#include <cstddef>
#include "../core/vector.hpp"

class DistanceMetric {
public:
    virtual float distance(const Vector& v1, const Vector& v2) const = 0;

    // Raw-pointer overload — avoids creating Vector objects (zero-copy from mmap)
    virtual float distance_raw(const float* a, const float* b, size_t dims) const = 0;

    virtual ~DistanceMetric() = default;
};

class EuclideanDistance : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
    float distance_raw(const float* a, const float* b, size_t dims) const override;
};

class ManhattanDistance : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
    float distance_raw(const float* a, const float* b, size_t dims) const override;
};

class CosineSimilarity : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
    float distance_raw(const float* a, const float* b, size_t dims) const override;
};
