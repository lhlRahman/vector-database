#include "lsh_index.hpp"
#include <algorithm>
#include <unordered_set>

LSHIndex::HashFunction::HashFunction(size_t dims) : random_vector(dims), bias(0.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for (size_t i = 0; i < dims; ++i) {
        random_vector[i] = d(gen);
    }
    bias = d(gen);
}

size_t LSHIndex::HashFunction::hash(const Vector& v) const {
    return Vector::simd_dot_product(v, random_vector) + bias > 0 ? 1 : 0;
}

LSHIndex::LSHIndex(size_t dimensions, size_t num_tables, size_t num_hash_functions)
    : num_tables(num_tables), num_hash_functions(num_hash_functions), dims(dimensions) {
    hash_tables.resize(num_tables);
    for (size_t i = 0; i < num_tables; ++i) {
        for (size_t j = 0; j < num_hash_functions; ++j) {
            hash_functions.emplace_back(dimensions);
        }
    }
}

void LSHIndex::insert(const Vector& vector, const std::string& key) {
    for (size_t i = 0; i < num_tables; ++i) {
        size_t hash = 0;
        for (size_t j = 0; j < num_hash_functions; ++j) {
            hash = (hash << 1) | hash_functions[i * num_hash_functions + j].hash(vector);
        }
        hash_tables[i][hash].emplace_back(vector, key);
    }
}

std::vector<std::pair<std::string, float>> LSHIndex::search(const Vector& query, size_t k, const DistanceMetric& metric) const {
    std::unordered_map<std::string, float> candidates;
    std::unordered_set<std::string> seen_keys;
    
    for (size_t i = 0; i < num_tables; ++i) {
        size_t hash = 0;
        for (size_t j = 0; j < num_hash_functions; ++j) {
            hash = (hash << 1) | hash_functions[i * num_hash_functions + j].hash(query);
        }
        
        auto it = hash_tables[i].find(hash);
        if (it != hash_tables[i].end()) {
            for (const auto& pair : it->second) {
                if (seen_keys.insert(pair.second).second) {
                    candidates[pair.second] = metric.distance(query, pair.first);
                }
            }
        }
    }
    
    std::vector<std::pair<std::string, float>> results(candidates.begin(), candidates.end());
    std::partial_sort(results.begin(), results.begin() + std::min(k, results.size()), results.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
    
    if (results.size() > k) {
        results.resize(k);
    }
    
    return results;
}