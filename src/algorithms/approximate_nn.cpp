// src/algorithms/approximate_nn.cpp

#include "approximate_nn.hpp"
#include "lsh_index.hpp"
#include "hnsw_index.hpp"
#include <random>
#include <unordered_set>
#include <algorithm>
#include <cmath>

#include "../optimizations/simd_operations.hpp"

RandomProjectionTrees::Node::Node(uint64_t sid, const std::string& k)
    : slot_id(sid), key(k), split_dimension(0) {}

RandomProjectionTrees::RandomProjectionTrees(size_t dimensions, size_t num_trees, size_t /*max_depth*/,
                                             VectorAccessor accessor,
                                             std::shared_ptr<const DistanceMetric> metric)
    : dimensions(dimensions), accessor_(std::move(accessor)),
      distance_metric(metric ? std::move(metric) : std::make_shared<EuclideanDistance>()) {
    trees.resize(num_trees);
}

void RandomProjectionTrees::insert(uint64_t slot_id, const std::string& key) {
    deleted_keys_.erase(key);
    for (auto& tree : trees) {
        insert_recursive(tree, slot_id, key, 0);
    }
}

void RandomProjectionTrees::remove(const std::string& key) {
    deleted_keys_.insert(key);
}

void RandomProjectionTrees::insert_recursive(std::unique_ptr<Node>& node, uint64_t slot_id, const std::string& key, size_t depth) {
    if (!node) {
        node = std::make_unique<Node>(slot_id, key);
        node->split_dimension = depth % dimensions;
        return;
    }

    size_t dim = depth % dimensions;
    const float* new_vec = accessor_(slot_id);
    const float* node_vec = accessor_(node->slot_id);
    if (new_vec[dim] < node_vec[dim]) {
        insert_recursive(node->left, slot_id, key, depth + 1);
    } else {
        insert_recursive(node->right, slot_id, key, depth + 1);
    }
}

std::vector<std::pair<std::string, float>> RandomProjectionTrees::search(const Vector& query, size_t k) const {
    std::vector<std::pair<std::string, float>> results;
    const float* q = query.data_ptr();
    for (const auto& tree : trees) {
        search_recursive(tree.get(), q, k, results);
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}

void RandomProjectionTrees::search_recursive(const Node* node, const float* query, size_t k,
                                              std::vector<std::pair<std::string, float>>& results) const {
    if (!node) return;

    if (deleted_keys_.count(node->key) > 0) {
        search_recursive(node->left.get(), query, k, results);
        search_recursive(node->right.get(), query, k, results);
        return;
    }

    const float* node_vec = accessor_(node->slot_id);
    float distance = distance_metric->distance_raw(query, node_vec, dimensions);
    results.emplace_back(node->key, distance);

    size_t dim = node->split_dimension;
    if (query[dim] < node_vec[dim]) {
        search_recursive(node->left.get(), query, k, results);
        if (results.size() < k || std::abs(query[dim] - node_vec[dim]) < results.back().second) {
            search_recursive(node->right.get(), query, k, results);
        }
    } else {
        search_recursive(node->right.get(), query, k, results);
        if (results.size() < k || std::abs(query[dim] - node_vec[dim]) < results.back().second) {
            search_recursive(node->left.get(), query, k, results);
        }
    }
}

std::unique_ptr<ApproximateNN> ApproximateNNFactory::create(const std::string& algorithm, size_t dimensions,
                                                            size_t param1, size_t param2,
                                                            std::shared_ptr<const DistanceMetric> metric,
                                                            VectorAccessor accessor) {
    if (algorithm == "LSH") {
        return std::make_unique<LSHIndex>(dimensions, param1, param2, metric, std::move(accessor));
    } else if (algorithm == "RPT") {
        return std::make_unique<RandomProjectionTrees>(dimensions, param1, param2, std::move(accessor), metric);
    } else if (algorithm == "HNSW") {
        return std::make_unique<HNSWIndex>(dimensions, param1, param2, param2, std::move(metric), std::move(accessor));
    }
    throw std::invalid_argument("Unknown algorithm: " + algorithm);
}
