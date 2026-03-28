// src/algorithms/approximate_nn.hpp

#pragma once

#include "../core/vector.hpp"
#include "../core/vector_accessor.hpp"
#include "../utils/distance_metrics.hpp"
#include <cstdint>
#include <string>
#include <memory>
#include <unordered_set>
#include <vector>

class ApproximateNN {
public:
    virtual ~ApproximateNN() = default;

    // Insert using mmap slot ID (zero-copy — vector data read via accessor)
    virtual void insert(uint64_t slot_id, const std::string& key) = 0;
    virtual void remove(const std::string& key) = 0;
    virtual std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const = 0;
};

class RandomProjectionTrees : public ApproximateNN {
private:
    struct Node {
        uint64_t slot_id;
        std::string key;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        size_t split_dimension;

        Node(uint64_t sid, const std::string& k);
    };

    std::vector<std::unique_ptr<Node>> trees;
    size_t dimensions;
    VectorAccessor accessor_;
    std::shared_ptr<const DistanceMetric> distance_metric;
    std::unordered_set<std::string> deleted_keys_;

    void insert_recursive(std::unique_ptr<Node>& node, uint64_t slot_id, const std::string& key, size_t depth);
    void search_recursive(const Node* node, const float* query, size_t k, std::vector<std::pair<std::string, float>>& results) const;

public:
    RandomProjectionTrees(size_t dimensions, size_t num_trees, size_t max_depth,
                          VectorAccessor accessor = nullptr,
                          std::shared_ptr<const DistanceMetric> metric = nullptr);

    void insert(uint64_t slot_id, const std::string& key) override;
    void remove(const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;
};

class ApproximateNNFactory {
public:
    static std::unique_ptr<ApproximateNN> create(const std::string& algorithm, size_t dimensions,
                                                  size_t param1, size_t param2,
                                                  std::shared_ptr<const DistanceMetric> metric,
                                                  VectorAccessor accessor);
};
