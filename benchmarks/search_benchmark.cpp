// benchmarks/search_benchmark.cpp
#include "../include/vector_database.hpp"
#include "../src/utils/random_generator.hpp"
#include <chrono>
#include <iostream>
#include <vector>

void benchmark_search(size_t dimensions, size_t num_vectors, size_t num_queries, size_t k) {
    VectorDatabase exact_db(dimensions, false);
    VectorDatabase approx_db(dimensions, true);
    RandomGenerator rng;

    // Insert vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v = rng.generateUniformVector(dimensions);
        exact_db.insert(v, "vector_" + std::to_string(i));
        approx_db.insert(v, "vector_" + std::to_string(i));
    }

    // Generate queries
    std::vector<Vector> queries;
    for (size_t i = 0; i < num_queries; ++i) {
        queries.push_back(rng.generateUniformVector(dimensions));
    }

    // Benchmark exact search
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries) {
        exact_db.similaritySearch(query, k);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto exact_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Benchmark approximate search
    start = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries) {
        approx_db.similaritySearch(query, k);
    }
    end = std::chrono::high_resolution_clock::now();
    auto approx_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Dimension: " << dimensions << ", Vectors: " << num_vectors << ", Queries: " << num_queries << ", k: " << k << std::endl;
    std::cout << "Exact search time: " << exact_duration.count() << " ms" << std::endl;
    std::cout << "Approximate search time: " << approx_duration.count() << " ms" << std::endl;
    std::cout << "Speedup: " << static_cast<double>(exact_duration.count()) / approx_duration.count() << "x" << std::endl << std::endl;
}

int main() {
    benchmark_search(128, 20, 1000, 10);
    return 0;
}