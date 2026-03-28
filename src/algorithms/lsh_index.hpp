
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "approximate_nn.hpp"
#include "../core/vector_accessor.hpp"

class LSHIndex : public ApproximateNN {
private:
    struct HashFunction {
        std::vector<float> random_vector; // raw float array instead of Vector
        float bias;

        explicit HashFunction(size_t dims);
        size_t hash(const float* v, size_t dims) const;
    };

    // hash_tables[table][hash_value] = vector of (slot_id, key)
    std::vector<std::vector<HashFunction>> hash_functions;
    std::vector<std::unordered_map<size_t, std::vector<std::pair<uint64_t, std::string>>>> hash_tables;
    size_t num_tables;
    size_t num_hash_functions;
    size_t dimensions;
    std::shared_ptr<const DistanceMetric> distance_metric;
    VectorAccessor accessor_;
    std::unordered_set<std::string> deleted_keys_;

public:
    LSHIndex(size_t dimensions, size_t num_tables, size_t num_hash_functions,
             std::shared_ptr<const DistanceMetric> metric, VectorAccessor accessor = nullptr);
    void insert(uint64_t slot_id, const std::string& key) override;
    void remove(const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;
};
