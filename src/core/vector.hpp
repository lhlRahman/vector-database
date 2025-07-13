// src/core/vector.hpp

#pragma once
#include <vector>
#include <cstddef>
#include <iostream>
#include <functional>

class Vector {
private:
    std::vector<float> data;

public:
    Vector() = default; // Default constructor
    Vector(size_t size);
    Vector(const std::vector<float>& values);
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    size_t size() const;
    const float* data_ptr() const;
    float* data_ptr();
    static float dot_product(const Vector& v1, const Vector& v2);

    bool operator==(const Vector& other) const {
        return data == other.data;
    }

    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    void write_to(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    }

    static Vector read_from(std::istream& is, size_t dimensions) {
        Vector v(dimensions);
        is.read(reinterpret_cast<char*>(v.data.data()), dimensions * sizeof(float));
        return v;
    }
};

// Specialize std::hash for Vector
namespace std {
    template <>
    struct hash<Vector> {
        size_t operator()(const Vector& v) const {
            size_t seed = 0;
            for (auto& i : v) {
                seed ^= std::hash<float>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
}
