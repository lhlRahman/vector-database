// src/algorithms/approximate_nn.cpp

#include "approximate_nn.hpp"
#include "lsh_index.hpp"
#include "hnsw_index.hpp"
#include <random>
#include <unordered_set>
#include <algorithm>

RandomProjectionTrees::Node::Node(const Vector& vec, const std::string& k)
    : vector(vec), key(k), split_dimension(-1) {}

RandomProjectionTrees::RandomProjectionTrees(size_t dimensions, size_t num_trees, size_t max_depth)
    : dimensions(dimensions), num_trees(num_trees), max_depth(max_depth) {
    trees.resize(num_trees);
}

void RandomProjectionTrees::insert(const Vector& vector, const std::string& key) {
    for (auto& tree : trees) {
        insert_recursive(tree, vector, key, 0);
    }
}

void RandomProjectionTrees::insert_recursive(std::unique_ptr<Node>& node, const Vector& vector, const std::string& key, int depth) {
    if (!node) {
        node = std::make_unique<Node>(vector, key);
        node->split_dimension = depth % dimensions;
        return;
    }

    int dim = depth % dimensions;
    if (vector[dim] < node->vector[dim]) {
        insert_recursive(node->left, vector, key, depth + 1);
    } else {
        insert_recursive(node->right, vector, key, depth + 1);
    }
}

std::vector<std::pair<std::string, float>> RandomProjectionTrees::search(const Vector& query, size_t k) const {
    std::vector<std::pair<std::string, float>> results;
    for (const auto& tree : trees) {
        search_recursive(tree.get(), query, k, results);
    }

    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    if (results.size() > k) {
        results.resize(k);
    }

    return results;
}

void RandomProjectionTrees::search_recursive(const Node* node, const Vector& query, size_t k, std::vector<std::pair<std::string, float>>& results) const {
    if (!node) {
        return;
    }

    float distance = Vector::dot_product(query, node->vector);
    results.emplace_back(node->key, distance);

    int dim = node->split_dimension;
    if (query[dim] < node->vector[dim]) {
        search_recursive(node->left.get(), query, k, results);
        if (results.size() < k || std::abs(query[dim] - node->vector[dim]) < results.back().second) {
            search_recursive(node->right.get(), query, k, results);
        }
    } else {
        search_recursive(node->right.get(), query, k, results);
        if (results.size() < k || std::abs(query[dim] - node->vector[dim]) < results.back().second) {
            search_recursive(node->left.get(), query, k, results);
        }
    }
}

std::unique_ptr<ApproximateNN> ApproximateNNFactory::create(const std::string& algorithm, size_t dimensions, size_t param1, size_t param2, const DistanceMetric* metric) {
    if (algorithm == "LSH") {
        return std::make_unique<LSHIndex>(dimensions, param1, param2, metric);
    } else if (algorithm == "RPT") {
        return std::make_unique<RandomProjectionTrees>(dimensions, param1, param2);
    } else if (algorithm == "HNSW") {
        return std::make_unique<HNSWIndex>(dimensions, param1, param2, param2, metric);
    }
    throw std::invalid_argument("Unknown algorithm: " + algorithm);
}