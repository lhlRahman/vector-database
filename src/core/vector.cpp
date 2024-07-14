#include "vector.hpp"
#include <stdexcept>

Vector::Vector(size_t size) : data(size) {}

Vector::Vector(const std::vector<float>& values) : data(values) {}

float& Vector::operator[](size_t index) {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

const float& Vector::operator[](size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

size_t Vector::size() const {
    return data.size();
}

const float* Vector::data_ptr() const {
    return data.data();
}

float Vector::simd_dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be the same size");
    }

    const float* a = v1.data_ptr();
    const float* b = v2.data_ptr();
    size_t size = v1.size();

    __m256 sum = _mm256_setzero_ps();

    for (size_t i = 0; i < size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    return result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
}