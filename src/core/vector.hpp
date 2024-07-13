#pragma once
#include <vector>
#include <cstddef>

class Vector {
    private:
        std::vector<float> data;
    public:
        Vector(size_t size);
        Vector(const std::vector<float>& values);

        float& operator[](size_t index);
        const float& operator[](size_t index) const; 

        size_t size() const;
};