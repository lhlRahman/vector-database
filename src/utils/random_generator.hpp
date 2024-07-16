// src/utils/random_generator.hpp
#pragma once
#include "../core/vector.hpp"
#include <random>

class RandomGenerator {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<float> uniform_dist;
    std::normal_distribution<float> normal_dist;

public:
    RandomGenerator(unsigned int seed = std::random_device{}());
    Vector generateUniformVector(size_t dimensions, float min = 0.0f, float max = 1.0f);
    Vector generateNormalVector(size_t dimensions, float mean = 0.0f, float stddev = 1.0f);
};