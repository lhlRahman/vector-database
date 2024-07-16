// src/algorithms/lsh_index.hpp

#pragma once

#include "approximate_nn.hpp"
#include <vector>
#include <unordered_map>

class LSHIndex : public ApproximateNN {
private:
    struct HashFunction {
        Vector random_vector;
        float bias;

        HashFunction(size_t dims);
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