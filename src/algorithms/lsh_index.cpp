
#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

#include "lsh_index.hpp"

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
    return Vector::dot_product(v, random_vector) + bias > 0 ? 1 : 0;
}

LSHIndex::LSHIndex(size_t dimensions, size_t num_tables, size_t num_hash_functions, const DistanceMetric* metric)
    : num_tables(num_tables), num_hash_functions(num_hash_functions), dims(dimensions), distance_metric(metric) {
    hash_tables.resize(num_tables);
    hash_functions.resize(num_tables, std::vector<HashFunction>(num_hash_functions, HashFunction(dimensions)));
}

void LSHIndex::insert(const Vector& vector, const std::string& key) {
    for (size_t i = 0; i < num_tables; ++i) {
        size_t hash = 0;
        for (size_t j = 0; j < num_hash_functions; ++j) {
            hash = (hash << 1) | hash_functions[i][j].hash(vector);
        }
        hash_tables[i][hash].emplace_back(vector, key);
    }
}

std::vector<std::pair<std::string, float>> LSHIndex::search(const Vector& query, size_t k) const {
    std::unordered_map<std::string, float> candidates;

    for (size_t i = 0; i < num_tables; ++i) {
        size_t hash = 0;
        for (size_t j = 0; j < num_hash_functions; ++j) {
            hash = (hash << 1) | hash_functions[i][j].hash(query);
        }

        auto it = hash_tables[i].find(hash);
        if (it != hash_tables[i].end()) {
            for (const auto& pair : it->second) {
                candidates[pair.second] = distance_metric->distance(query, pair.first);
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