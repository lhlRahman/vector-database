
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
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (size_t i = 0; i < dims; ++i) {
        random_vector[i] = d(gen);
    }
    bias = d(gen);
}

size_t LSHIndex::HashFunction::hash(const Vector& v) const {
    return Vector::dot_product(v, random_vector) + bias > 0 ? 1 : 0;
}

LSHIndex::LSHIndex(size_t dimensions, size_t num_tables, size_t num_hash_functions, std::shared_ptr<const DistanceMetric> metric)
    : num_tables(num_tables), num_hash_functions(num_hash_functions), distance_metric(std::move(metric)) {
    hash_tables.resize(num_tables);
    hash_functions.resize(num_tables);
    for (size_t t = 0; t < num_tables; ++t) {
        hash_functions[t].reserve(num_hash_functions);
        for (size_t h = 0; h < num_hash_functions; ++h) {
            hash_functions[t].emplace_back(dimensions);
        }
    }
}

void LSHIndex::insert(const Vector& vector, const std::string& key) {
    deleted_keys_.erase(key);
    for (size_t i = 0; i < num_tables; ++i) {
        size_t hash = 0;
        for (size_t j = 0; j < num_hash_functions; ++j) {
            hash = (hash << 1) | hash_functions[i][j].hash(vector);
        }
        hash_tables[i][hash].emplace_back(vector, key);
    }
}

void LSHIndex::remove(const std::string& key) {
    deleted_keys_.insert(key);
}

std::vector<std::pair<std::string, float>> LSHIndex::search(const Vector& query, size_t k) const {
    std::unordered_map<std::string, float> candidates;
    candidates.reserve(k * num_tables); // Pre-reserve to avoid rehashes

    for (size_t i = 0; i < num_tables; ++i) {
        size_t hash = 0;
        for (size_t j = 0; j < num_hash_functions; ++j) {
            hash = (hash << 1) | hash_functions[i][j].hash(query);
        }

        auto it = hash_tables[i].find(hash);
        if (it != hash_tables[i].end()) {
            for (const auto& pair : it->second) {
                if (deleted_keys_.count(pair.second) == 0) {
                    candidates[pair.second] = distance_metric->distance(query, pair.first);
                }
            }
        }
    }

    std::vector<std::pair<std::string, float>> results(candidates.begin(), candidates.end());
    size_t actual_k = std::min(k, results.size());
    std::partial_sort(results.begin(),
                      results.begin() + static_cast<std::ptrdiff_t>(actual_k),
                      results.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

    results.resize(actual_k);

    return results;
}