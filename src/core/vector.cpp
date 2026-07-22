
#include <stdexcept>
#include <vector>
#include "vector.hpp"
#include "../optimizations/simd_operations.hpp"

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
    return simd_ops::dot_product(v1, v2);
}

void Vector::enable_simd(bool enable) {
    simd_ops::set_enabled(enable);
}

bool Vector::is_simd_enabled() {
    return simd_ops::is_enabled();
}

void Vector::write_to(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(float)));
}

Vector Vector::read_from(std::istream& is, size_t dimensions) {
    Vector v(dimensions);
    is.read(reinterpret_cast<char*>(v.data.data()),
            static_cast<std::streamsize>(dimensions * sizeof(float)));
    return v;
}
