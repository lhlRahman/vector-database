#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vector.hpp"
#include "vector_accessor.hpp"
#include "../utils/distance_metrics.hpp"

class KDTree {
private:
    struct Node {
        uint64_t slot_id;
        std::string key;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        size_t split_dimension;

        Node(uint64_t sid, std::string k);
    };

    std::unique_ptr<Node> root;
    size_t dimensions;
    std::shared_ptr<DistanceMetric> distanceMetric;
    VectorAccessor accessor_;
    std::unordered_map<std::string, uint64_t> key_to_slot_;
    std::unordered_set<std::string> temporarilyRemoved;

    void insert_recursive(std::unique_ptr<Node>& node, uint64_t slot_id, const std::string& key, size_t depth);
    void nearest_neighbor_recursive(const Node* node, const float* query, std::string& best_key, float& best_distance, size_t depth) const;

public:
    KDTree(size_t dimensions, std::shared_ptr<DistanceMetric> metric, VectorAccessor accessor);
    void insert(uint64_t slot_id, const std::string& key);
    void remove(const std::string& key);
    std::string nearest_neighbor(const Vector& query) const;
    std::vector<std::pair<std::string, float>> nearestNeighbors(const Vector& query, size_t k) const;
    void removeTemporarily(const std::string& key);
    void reinsert(const std::string& key);
};
