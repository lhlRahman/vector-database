// src/core/vector.cpp

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

float* Vector::data_ptr() {
    return data.data();
}

float Vector::dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be the same size");
    }
    float result = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}
