
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kd_tree.hpp"

KDTree::Node::Node(Vector vec, std::string k) : vector(std::move(vec)), key(std::move(k)), split_dimension(0) {}

KDTree::KDTree(size_t dims, std::shared_ptr<DistanceMetric> metric) 
    : dimensions(dims), distanceMetric(std::move(metric)) {}

void KDTree::insert(const Vector& vector, const std::string& key) {
    temporarilyRemoved.erase(key);
    vectorMap[key] = vector;
    insert_recursive(root, vector, key, 0);
}

void KDTree::remove(const std::string& key) {
    vectorMap.erase(key);
    temporarilyRemoved.insert(key);
}

void KDTree::insert_recursive(std::unique_ptr<Node>& node, const Vector& vector, const std::string& key, size_t depth) {
    if (!node) {
        node = std::make_unique<Node>(vector, key);
        node->split_dimension = depth % dimensions;
        return;
    }

    size_t dim = depth % dimensions;
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

void KDTree::nearest_neighbor_recursive(const Node* node, const Vector& query, std::string& best_key, float& best_distance, size_t depth) const {
    if (!node || (!temporarilyRemoved.empty() && temporarilyRemoved.count(node->key) > 0)) {
        return;
    }

    float distance = distanceMetric->distance(query, node->vector);
    if (distance < best_distance) {
        best_distance = distance;
        best_key = node->key;
    }

    size_t dim = depth % dimensions;
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
    std::vector<std::pair<std::string, float>> distances;
    distances.reserve(vectorMap.size());

    for (const auto& [key, vec] : vectorMap) {
        if (!temporarilyRemoved.empty() && temporarilyRemoved.count(key) > 0) {
            continue;
        }
        distances.emplace_back(key, distanceMetric->distance(query, vec));
    }

    if (distances.empty()) {
        return {};
    }

    size_t actual_k = std::min(k, distances.size());
    std::partial_sort(distances.begin(),
                      distances.begin() + static_cast<std::ptrdiff_t>(actual_k),
                      distances.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

    distances.resize(actual_k);
    return distances;
}
