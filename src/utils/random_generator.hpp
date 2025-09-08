#pragma once

#include <random>

#include "../core/vector.hpp"

class RandomGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform_dist;
    std::normal_distribution<float> normal_dist;

public:
    explicit RandomGenerator(unsigned int seed = std::random_device{}()); // Marked explicit
    Vector generateUniformVector(size_t dimensions, float min = 0.0f, float max = 1.0f);
    Vector generateNormalVector(size_t dimensions, float mean = 0.0f, float stddev = 1.0f);
};