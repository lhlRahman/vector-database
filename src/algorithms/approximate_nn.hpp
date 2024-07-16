// src/algorithms/approximate_nn.hpp

#pragma once

#include "../core/vector.hpp"
#include "../utils/distance_metrics.hpp"
#include <vector>
#include <string>
#include <memory>

class ApproximateNN {
public:
    virtual ~ApproximateNN() = default;

    virtual void insert(const Vector& vector, const std::string& key) = 0;
    virtual std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const = 0;
};

class RandomProjectionTrees : public ApproximateNN {
private:
    struct Node {
        Vector vector;
        std::string key;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        int split_dimension;

        Node(const Vector& vec, const std::string& k);
    };

    std::vector<std::unique_ptr<Node>> trees;
    size_t dimensions;
    size_t num_trees;
    size_t max_depth;

    void insert_recursive(std::unique_ptr<Node>& node, const Vector& vector, const std::string& key, int depth);
    void search_recursive(const Node* node, const Vector& query, size_t k, std::vector<std::pair<std::string, float>>& results) const;

public:
    RandomProjectionTrees(size_t dimensions, size_t num_trees, size_t max_depth);

    void insert(const Vector& vector, const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;
};

class ApproximateNNFactory {
public:
    static std::unique_ptr<ApproximateNN> create(const std::string& algorithm, size_t dimensions, size_t param1, size_t param2, const DistanceMetric* metric);
};