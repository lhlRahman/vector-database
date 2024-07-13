#include "vector.hpp"
#include <stdexcept>

Vector::Vector(size_t size) : data(size) {}

Vector::Vector(const std::vector<float>& values) : data(values) {}

// overloads [] so it checks if inbounds and returns the value
float& Vector::operator[](size_t index) {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

// overloads [] so it checks if inbounds and returns the value
const float& Vector::operator[](size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

size_t Vector::size() const {
    return data.size();
}