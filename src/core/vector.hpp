
#pragma once
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

class Vector {
private:
    std::vector<float> data;

public:
    Vector() = default;
    explicit Vector(size_t size);
    explicit Vector(const std::vector<float>& values);
    explicit Vector(std::vector<float>&& values) noexcept : data(std::move(values)) {}

    // Defaulted moves so that vector<Vector> growth is noexcept-movable.
    Vector(const Vector&)                = default;
    Vector(Vector&&) noexcept            = default;
    Vector& operator=(const Vector&)     = default;
    Vector& operator=(Vector&&) noexcept = default;
    ~Vector()                            = default;

    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    size_t size() const;
    const float* data_ptr() const;
    float* data_ptr();
    static float dot_product(const Vector& v1, const Vector& v2);
    
    // SIMD control
    static void enable_simd(bool enable);
    static bool is_simd_enabled();

    bool operator==(const Vector& other) const {
        return data == other.data;
    }

    auto begin() const { return data.cbegin(); }
    auto end() const { return data.cend(); }

    void write_to(std::ostream& os) const;
    static Vector read_from(std::istream& is, size_t dimensions);
};

// Specialize std::hash for Vector. Boost-style hash combine.
namespace std {
    template <>
    struct hash<Vector> {
        size_t operator()(const Vector& v) const noexcept {
            size_t seed = 0;
            for (const float f : v) {
                seed ^= std::hash<float>{}(f) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
} // namespace std
