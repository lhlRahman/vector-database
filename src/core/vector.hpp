#pragma once
#include <vector>
#include <cstddef>
#include <immintrin.h>

class Vector {
private:
    std::vector<float> data;

public:
    Vector(size_t size);
    Vector(const std::vector<float>& values);
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    size_t size() const;
    const float* data_ptr() const;
    static float simd_dot_product(const Vector& v1, const Vector& v2);
};
