// src/utils/distance_metrics.hpp
#pragma once
#include "../core/vector.hpp"

class DistanceMetric {
public:
    virtual float distance(const Vector& v1, const Vector& v2) const = 0;
    virtual ~DistanceMetric() = default;
};

class EuclideanDistance : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
};

class ManhattanDistance : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
};

class CosineSimilarity : public DistanceMetric {
public:
    float distance(const Vector& v1, const Vector& v2) const override;
};