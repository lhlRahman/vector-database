/**
 * SIMD tail handling test
 * Validates dot/add/subtract for non-multiple vector sizes.
 */

#include <iostream>
#include <string>
#include "../src/core/vector.hpp"
#include "../src/optimizations/simd_operations.hpp"

int main() {
    const size_t dims = 130; // Not divisible by 4 or 8
    Vector a(dims);
    Vector b(dims);
    Vector sum(dims);
    Vector diff(dims);

    for (size_t i = 0; i < dims; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    const float dot = simd_ops::dot_product(a, b);
    const float expected_dot = 2.0f * static_cast<float>(dims);
    if (dot != expected_dot) {
        std::cerr << "dot_product failed: expected " << expected_dot << ", got " << dot << std::endl;
        return 1;
    }

    simd_ops::add(a, b, sum);
    simd_ops::subtract(b, a, diff);

    for (size_t i = 0; i < dims; ++i) {
        if (sum[i] != 3.0f) {
            std::cerr << "add failed at index " << i << ": " << sum[i] << std::endl;
            return 1;
        }
        if (diff[i] != 1.0f) {
            std::cerr << "subtract failed at index " << i << ": " << diff[i] << std::endl;
            return 1;
        }
    }

    std::cout << "SIMD tail test passed for dims=" << dims << std::endl;
    return 0;
}
