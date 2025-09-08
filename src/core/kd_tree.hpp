#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vector.hpp"
#include "../utils/distance_metrics.hpp"

class KDTree {
private:
    struct Node {
        Vector vector;
        std::string key;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        int split_dimension;

        Node(const Vector& vec, const std::string& k);
    };

    std::unique_ptr<Node> root;
    size_t dimensions;
    std::shared_ptr<DistanceMetric> distanceMetric;
    std::unordered_map<std::string, Vector> vectorMap;
    std::unordered_set<std::string> temporarilyRemoved;

    void insert_recursive(std::unique_ptr<Node>& node, const Vector& vector, const std::string& key, int depth);
    void nearest_neighbor_recursive(const Node* node, const Vector& query, std::string& best_key, float& best_distance, int depth) const;

public:
    KDTree(size_t dimensions, std::shared_ptr<DistanceMetric> metric);
    void insert(const Vector& vector, const std::string& key);
    std::string nearest_neighbor(const Vector& query) const;
    std::vector<std::pair<std::string, float>> nearestNeighbors(const Vector& query, size_t k) const;
    const Vector& getVector(const std::string& key) const;
    void removeTemporarily(const std::string& key);
    void reinsert(const std::string& key);
};