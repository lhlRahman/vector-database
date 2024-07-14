// lsh_index.hpp
#pragma once
#include "vector.hpp"
#include "distance_metrics.hpp"
#include <unordered_map>
#include <random>
#include <vector>
#include <string>

class LSHIndex {
private:
    struct HashFunction {
        Vector random_vector;
        float bias;
        
        HashFunction(size_t dims);
        size_t hash(const Vector& v) const;
    };
    
    std::vector<HashFunction> hash_functions;
    std::vector<std::unordered_map<size_t, std::vector<std::pair<Vector, std::string>>>> hash_tables;
    size_t num_tables;
    size_t num_hash_functions;
    size_t dims;
    
public:
    LSHIndex(size_t dimensions, size_t num_tables, size_t num_hash_functions);
    void insert(const Vector& vector, const std::string& key);
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k, const DistanceMetric& metric) const;
};