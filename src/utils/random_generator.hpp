#pragma once

#include <random>

#include "../core/vector.hpp"

class RandomGenerator {
private:
    // Only the engine is retained. generate*Vector() build a fresh distribution
    // per call (they take min/max/mean/stddev), so member distributions would be
    // dead state whose parameters silently had no effect.
    std::mt19937 gen;

public:
    explicit RandomGenerator(unsigned int seed = std::random_device{}()); // Marked explicit
    Vector generateUniformVector(size_t dimensions, float min = 0.0f, float max = 1.0f);
    Vector generateNormalVector(size_t dimensions, float mean = 0.0f, float stddev = 1.0f);
};