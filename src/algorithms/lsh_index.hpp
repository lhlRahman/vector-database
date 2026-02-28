
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    std::shared_ptr<const DistanceMetric> distance_metric;
    std::unordered_set<std::string> deleted_keys_;

public:
    LSHIndex(size_t dimensions, size_t num_tables, size_t num_hash_functions, std::shared_ptr<const DistanceMetric> metric);
    void insert(const Vector& vector, const std::string& key) override;
    void remove(const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;
};