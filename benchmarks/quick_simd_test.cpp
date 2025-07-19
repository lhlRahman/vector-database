// benchmarks/quick_simd_test.cpp
#include "../include/vector_database.hpp"
#include "../src/optimizations/simd_operations.hpp"
#include "../src/utils/random_generator.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

// Simple scalar dot product for comparison
float scalar_dot_product(const Vector& v1, const Vector& v2) {
    float result = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

int main() {
    std::cout << "🚀 SIMD Performance Test on Apple Silicon (M1)" << std::endl;
    std::cout << "===============================================" << std::endl << std::endl;
    
    RandomGenerator rng;
    const size_t dimensions = 128;
    const size_t num_tests = 50000;
    
    // Generate test vectors
    Vector v1 = rng.generateUniformVector(dimensions);
    Vector v2 = rng.generateUniformVector(dimensions);
    
    // Test scalar performance
    auto start = std::chrono::high_resolution_clock::now();
    float scalar_result = 0.0f;
    for (size_t i = 0; i < num_tests; ++i) {
        scalar_result = scalar_dot_product(v1, v2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test SIMD performance
    start = std::chrono::high_resolution_clock::now();
    float simd_result = 0.0f;
    for (size_t i = 0; i < num_tests; ++i) {
        simd_result = simd_ops::dot_product(v1, v2);
    }
    end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate speedup
    double speedup = static_cast<double>(scalar_time.count()) / simd_time.count();
    
    // Check if results match (allowing for floating point precision)
    bool results_match = std::abs(scalar_result - simd_result) < 1e-6;
    
    std::cout << "📊 Results Summary:" << std::endl;
    std::cout << "   Vector dimensions: " << dimensions << std::endl;
    std::cout << "   Number of operations: " << num_tests << std::endl;
    std::cout << "   Scalar time: " << scalar_time.count() << " μs" << std::endl;
    std::cout << "   SIMD time: " << simd_time.count() << " μs" << std::endl;
    std::cout << "   Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "   Results match: " << (results_match ? "✅" : "❌") << std::endl;
    std::cout << std::endl;
    
    if (speedup > 2.0) {
        std::cout << "🎉 Excellent! SIMD is providing significant performance improvement!" << std::endl;
    } else if (speedup > 1.5) {
        std::cout << "👍 Good! SIMD is providing noticeable performance improvement." << std::endl;
    } else {
        std::cout << "⚠️  SIMD improvement is minimal. This might be due to small vector sizes or overhead." << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "💡 Note: The speedup you see depends on:" << std::endl;
    std::cout << "   - Vector size (larger vectors = better SIMD performance)" << std::endl;
    std::cout << "   - CPU architecture (ARM NEON vs x86 AVX)" << std::endl;
    std::cout << "   - Compiler optimizations" << std::endl;
    
    return 0;
} 