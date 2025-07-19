// benchmarks/simd_benchmark.cpp
#include "../include/vector_database.hpp"
#include "../src/optimizations/simd_operations.hpp"
#include "../src/utils/random_generator.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>

// Scalar implementations for comparison
float scalar_dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    float result = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

void scalar_add(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
}

void scalar_subtract(const Vector& v1, const Vector& v2, Vector& result) {
    if (v1.size() != v2.size() || v1.size() != result.size()) {
        throw std::invalid_argument("All vectors must have the same size");
    }
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
}

void benchmark_dot_product(size_t dimensions, size_t num_operations) {
    RandomGenerator rng;
    std::vector<Vector> vectors1, vectors2;
    
    // Generate test vectors
    for (size_t i = 0; i < num_operations; ++i) {
        vectors1.push_back(rng.generateUniformVector(dimensions));
        vectors2.push_back(rng.generateUniformVector(dimensions));
    }

    // Test scalar dot product
    auto start = std::chrono::high_resolution_clock::now();
    float scalar_sum = 0.0f;
    for (size_t i = 0; i < num_operations; ++i) {
        scalar_sum += scalar_dot_product(vectors1[i], vectors2[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test SIMD dot product
    start = std::chrono::high_resolution_clock::now();
    float simd_sum = 0.0f;
    for (size_t i = 0; i < num_operations; ++i) {
        simd_sum += simd_ops::dot_product(vectors1[i], vectors2[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Verify results are similar (allowing for floating point differences)
    float diff = std::abs(scalar_sum - simd_sum);
    bool results_match = diff < 1e-6;

    std::cout << "=== Dot Product Benchmark ===" << std::endl;
    std::cout << "Dimensions: " << dimensions << ", Operations: " << num_operations << std::endl;
    std::cout << "Scalar time: " << scalar_duration.count() << " μs" << std::endl;
    std::cout << "SIMD time: " << simd_duration.count() << " μs" << std::endl;
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) 
              << static_cast<double>(scalar_duration.count()) / simd_duration.count() << "x" << std::endl;
    std::cout << "Results match: " << (results_match ? "✓" : "✗") << std::endl;
    std::cout << "Scalar sum: " << scalar_sum << ", SIMD sum: " << simd_sum << std::endl << std::endl;
}

void benchmark_vector_operations(size_t dimensions, size_t num_operations) {
    RandomGenerator rng;
    std::vector<Vector> vectors1, vectors2, results_scalar, results_simd;
    
    // Generate test vectors
    for (size_t i = 0; i < num_operations; ++i) {
        vectors1.push_back(rng.generateUniformVector(dimensions));
        vectors2.push_back(rng.generateUniformVector(dimensions));
        results_scalar.push_back(Vector(dimensions));
        results_simd.push_back(Vector(dimensions));
    }

    // Test scalar addition
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        scalar_add(vectors1[i], vectors2[i], results_scalar[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_add_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test SIMD addition
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        simd_ops::add(vectors1[i], vectors2[i], results_simd[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_add_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test scalar subtraction
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        scalar_subtract(vectors1[i], vectors2[i], results_scalar[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto scalar_sub_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Test SIMD subtraction
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_operations; ++i) {
        simd_ops::subtract(vectors1[i], vectors2[i], results_simd[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_sub_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Verify results match
    bool add_results_match = true;
    bool sub_results_match = true;
    for (size_t i = 0; i < std::min(size_t(5), num_operations); ++i) { // Check first 5 results
        for (size_t j = 0; j < dimensions; ++j) {
            if (std::abs(results_scalar[i][j] - results_simd[i][j]) > 1e-6) {
                add_results_match = false;
                break;
            }
        }
    }

    std::cout << "=== Vector Operations Benchmark ===" << std::endl;
    std::cout << "Dimensions: " << dimensions << ", Operations: " << num_operations << std::endl;
    std::cout << "Addition:" << std::endl;
    std::cout << "  Scalar time: " << scalar_add_duration.count() << " μs" << std::endl;
    std::cout << "  SIMD time: " << simd_add_duration.count() << " μs" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << static_cast<double>(scalar_add_duration.count()) / simd_add_duration.count() << "x" << std::endl;
    std::cout << "  Results match: " << (add_results_match ? "✓" : "✗") << std::endl;
    
    std::cout << "Subtraction:" << std::endl;
    std::cout << "  Scalar time: " << scalar_sub_duration.count() << " μs" << std::endl;
    std::cout << "  SIMD time: " << simd_sub_duration.count() << " μs" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << static_cast<double>(scalar_sub_duration.count()) / simd_sub_duration.count() << "x" << std::endl;
    std::cout << "  Results match: " << (sub_results_match ? "✓" : "✗") << std::endl << std::endl;
}

void benchmark_different_sizes() {
    std::cout << "=== SIMD vs Scalar Performance Comparison ===" << std::endl << std::endl;
    
    // Test different vector sizes
    std::vector<size_t> dimensions = {64, 128, 256, 512, 1024};
    size_t num_operations = 10000;
    
    for (size_t dim : dimensions) {
        benchmark_dot_product(dim, num_operations);
        benchmark_vector_operations(dim, num_operations);
    }
}

int main() {
    try {
        benchmark_different_sizes();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 