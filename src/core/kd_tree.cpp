
#include <algorithm>
#include <limits>
#include <memory>      // Added for std::make_unique
#include <string>      // Added for std::string
#include <unordered_set> // Added for std::unordered_set
#include <utility>     // Added for std::move
#include <vector>      // Added for std::vector

#include "kd_tree.hpp"

KDTree::Node::Node(const Vector& vec, const std::string& k) : vector(vec), key(k), split_dimension(-1) {}

KDTree::KDTree(size_t dims, std::shared_ptr<DistanceMetric> metric) 
    : dimensions(dims), distanceMetric(std::move(metric)) {}

void KDTree::insert(const Vector& vector, const std::string& key) {
    vectorMap[key] = vector;
    insert_recursive(root, vector, key, 0);
}

void KDTree::insert_recursive(std::unique_ptr<Node>& node, const Vector& vector, const std::string& key, int depth) {
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

std::string KDTree::nearest_neighbor(const Vector& query) const {
    if (!root) {
        return "";
    }
    std::string best_key = root->key;
    float best_distance = distanceMetric->distance(query, root->vector);
    nearest_neighbor_recursive(root.get(), query, best_key, best_distance, 0);
    return best_key;
}

void KDTree::nearest_neighbor_recursive(const Node* node, const Vector& query, std::string& best_key, float& best_distance, int depth) const {
    if (!node || temporarilyRemoved.count(node->key) > 0) {
        return;
    }

    float distance = distanceMetric->distance(query, node->vector);
    if (distance < best_distance) {
        best_distance = distance;
        best_key = node->key;
    }

    int dim = depth % dimensions;
    float delta = query[dim] - node->vector[dim];
    const Node* first = delta < 0 ? node->left.get() : node->right.get();
    const Node* second = delta < 0 ? node->right.get() : node->left.get();

    nearest_neighbor_recursive(first, query, best_key, best_distance, depth + 1);

    if (delta * delta < best_distance) {
        nearest_neighbor_recursive(second, query, best_key, best_distance, depth + 1);
    }
}

const Vector& KDTree::getVector(const std::string& key) const {
    return vectorMap.at(key);
}

void KDTree::removeTemporarily(const std::string& key) {
    temporarilyRemoved.insert(key);
}

void KDTree::reinsert(const std::string& key) {
    temporarilyRemoved.erase(key);
}

std::vector<std::pair<std::string, float>> KDTree::nearestNeighbors(const Vector& query, size_t k) const {
    std::vector<std::pair<std::string, float>> result;
    for (size_t i = 0; i < k; ++i) {
        std::string nearest = nearest_neighbor(query);
        float distance = distanceMetric->distance(query, getVector(nearest));
        result.emplace_back(nearest, distance);
        const_cast<KDTree*>(this)->removeTemporarily(nearest);
    }
    for (const auto& pair : result) {
        const_cast<KDTree*>(this)->reinsert(pair.first);
    }
    return result;
}
