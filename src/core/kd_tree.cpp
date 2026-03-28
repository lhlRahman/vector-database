
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kd_tree.hpp"

KDTree::Node::Node(uint64_t sid, std::string k) : slot_id(sid), key(std::move(k)), split_dimension(0) {}

KDTree::KDTree(size_t dims, std::shared_ptr<DistanceMetric> metric, VectorAccessor accessor)
    : dimensions(dims), distanceMetric(std::move(metric)), accessor_(std::move(accessor)) {}

void KDTree::insert(uint64_t slot_id, const std::string& key) {
    temporarilyRemoved.erase(key);
    key_to_slot_[key] = slot_id;
    insert_recursive(root, slot_id, key, 0);
}

void KDTree::remove(const std::string& key) {
    key_to_slot_.erase(key);
    temporarilyRemoved.insert(key);
}

void KDTree::insert_recursive(std::unique_ptr<Node>& node, uint64_t slot_id, const std::string& key, size_t depth) {
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

std::string KDTree::nearest_neighbor(const Vector& query) const {
    if (!root) {
        return "";
    }
    const float* q = query.data_ptr();
    std::string best_key = root->key;
    float best_distance = distanceMetric->distance_raw(q, accessor_(root->slot_id), dimensions);
    nearest_neighbor_recursive(root.get(), q, best_key, best_distance, 0);
    return best_key;
}

void KDTree::nearest_neighbor_recursive(const Node* node, const float* query, std::string& best_key, float& best_distance, size_t depth) const {
    if (!node || (!temporarilyRemoved.empty() && temporarilyRemoved.count(node->key) > 0)) {
        return;
    }

    const float* node_vec = accessor_(node->slot_id);
    float distance = distanceMetric->distance_raw(query, node_vec, dimensions);
    if (distance < best_distance) {
        best_distance = distance;
        best_key = node->key;
    }

    size_t dim = depth % dimensions;
    float delta = query[dim] - node_vec[dim];
    const Node* first = delta < 0 ? node->left.get() : node->right.get();
    const Node* second = delta < 0 ? node->right.get() : node->left.get();

    nearest_neighbor_recursive(first, query, best_key, best_distance, depth + 1);

    if (delta * delta < best_distance) {
        nearest_neighbor_recursive(second, query, best_key, best_distance, depth + 1);
    }
}

void KDTree::removeTemporarily(const std::string& key) {
    temporarilyRemoved.insert(key);
}

void KDTree::reinsert(const std::string& key) {
    temporarilyRemoved.erase(key);
}

std::vector<std::pair<std::string, float>> KDTree::nearestNeighbors(const Vector& query, size_t k) const {
    const float* q = query.data_ptr();

    // Collect candidate pointers for batched distance with prefetch
    std::vector<std::pair<std::string, uint64_t>> active_entries;
    active_entries.reserve(key_to_slot_.size());
    for (const auto& [key, slot_id] : key_to_slot_) {
        if (!temporarilyRemoved.empty() && temporarilyRemoved.count(key) > 0) continue;
        active_entries.emplace_back(key, slot_id);
    }

    if (active_entries.empty()) return {};

    // Batch distance computation with prefetch
    std::vector<std::pair<std::string, float>> distances;
    distances.reserve(active_entries.size());

    for (size_t i = 0; i < active_entries.size(); ++i) {
        // Prefetch next vector from mmap
        if (i + 1 < active_entries.size()) {
            __builtin_prefetch(accessor_(active_entries[i + 1].second), 0, 1);
        }
        const float* vec = accessor_(active_entries[i].second);
        distances.emplace_back(active_entries[i].first,
                               distanceMetric->distance_raw(q, vec, dimensions));
    }

    size_t actual_k = std::min(k, distances.size());
    std::partial_sort(distances.begin(),
                      distances.begin() + static_cast<std::ptrdiff_t>(actual_k),
                      distances.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

    distances.resize(actual_k);
    return distances;
}
