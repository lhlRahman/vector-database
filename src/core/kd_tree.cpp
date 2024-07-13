#include "kd_tree.hpp"
#include "../utils/distance_metrics.hpp"
#include <limits>
#include <algorithm>

KDTree::KDTree(size_t dims) : dimensions(dims) {}

void KDTree::insert(const Vector& vector, const std::string& key) {
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
    float best_distance = DistanceMetrics::euclidean(query, root->vector);
    nearest_neighbor_recursive(root.get(), query, best_key, best_distance, 0);
    return best_key;
}

void KDTree::nearest_neighbor_recursive(const Node* node, const Vector& query, std::string& best_key, float& best_distance, int depth) const {
    if (!node) {
        return;
    }

    float distance = DistanceMetrics::euclidean(query, node->vector);
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

