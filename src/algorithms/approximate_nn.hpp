// src/algorithms/approximate_nn.hpp

#pragma once

#include "../core/vector.hpp"
#include "../utils/distance_metrics.hpp"
#include <string>
#include <memory>
#include <unordered_set>
#include <vector>

class ApproximateNN {
public:
    virtual ~ApproximateNN() = default;

    virtual void insert(const Vector& vector, const std::string& key) = 0;
    virtual void remove(const std::string& key) = 0;
    virtual std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const = 0;
};

class RandomProjectionTrees : public ApproximateNN {
private:
    struct Node {
        Vector vector;
        std::string key;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        size_t split_dimension;

        Node(const Vector& vec, const std::string& k);
    };

    std::vector<std::unique_ptr<Node>> trees;
    size_t dimensions;
    std::unordered_set<std::string> deleted_keys_;

    void insert_recursive(std::unique_ptr<Node>& node, const Vector& vector, const std::string& key, size_t depth);
    void search_recursive(const Node* node, const Vector& query, size_t k, std::vector<std::pair<std::string, float>>& results) const;

public:
    RandomProjectionTrees(size_t dimensions, size_t num_trees, size_t max_depth);

    void insert(const Vector& vector, const std::string& key) override;
    void remove(const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;
};

class ApproximateNNFactory {
public:
    static std::unique_ptr<ApproximateNN> create(const std::string& algorithm, size_t dimensions, size_t param1, size_t param2, std::shared_ptr<const DistanceMetric> metric);
};