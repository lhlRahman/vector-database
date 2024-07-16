// benchmarks/insertion_benchmark.cpp
#include "../include/vector_database.hpp"
#include "../src/utils/random_generator.hpp"
#include <chrono>
#include <iostream>
#include <vector>

void benchmark_insertion(size_t dimensions, size_t num_vectors) {
    VectorDatabase db(dimensions);
    RandomGenerator rng;

    std::vector<Vector> vectors;
    std::vector<std::string> keys;

    // Generate vectors and keys
    for (size_t i = 0; i < num_vectors; ++i) {
        vectors.push_back(rng.generateUniformVector(dimensions));
        keys.push_back("vector_" + std::to_string(i));
    }

    // Benchmark insertion
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_vectors; ++i) {
        db.insert(vectors[i], keys[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Inserted " << num_vectors << " vectors of dimension " << dimensions << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average time per insertion: " << static_cast<double>(duration.count()) / num_vectors << " ms" << std::endl;
}

int main() {
    benchmark_insertion(128, 100000);
    benchmark_insertion(256, 100000);
    benchmark_insertion(512, 100000);
    return 0;
}