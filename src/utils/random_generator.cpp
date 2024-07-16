// src/utils/random_generator.cpp

#include "random_generator.hpp"

RandomGenerator::RandomGenerator(unsigned int seed) : gen(seed), uniform_dist(0.0f, 1.0f), normal_dist(0.0f, 1.0f) {}

Vector RandomGenerator::generateUniformVector(size_t dimensions, float min, float max) {
    Vector v(dimensions);
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < dimensions; ++i) {
        v[i] = dist(gen);
    }
    return v;
}

Vector RandomGenerator::generateNormalVector(size_t dimensions, float mean, float stddev) {
    Vector v(dimensions);
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < dimensions; ++i) {
        v[i] = dist(gen);
    }
    return v;
}