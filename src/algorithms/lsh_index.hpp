
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "approximate_nn.hpp"

class LSHIndex : public ApproximateNN {
private:
    struct HashFunction {
        Vector random_vector;
        float bias;

        explicit HashFunction(size_t dims); // Marked explicit
        size_t hash(const Vector& v) const;
    };

    std::vector<std::vector<HashFunction>> hash_functions;
    std::vector<std::unordered_map<size_t, std::vector<std::pair<Vector, std::string>>>> hash_tables;
    size_t num_tables;
    size_t num_hash_functions;
    size_t dims;
    const DistanceMetric* distance_metric;

public:
    LSHIndex(size_t dimensions, size_t num_tables, size_t num_hash_functions, const DistanceMetric* metric);
    void insert(const Vector& vector, const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;
};