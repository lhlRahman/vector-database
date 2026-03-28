
#pragma once

#include <cstdint>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "approximate_nn.hpp"
#include "../core/vector.hpp"
#include "../core/vector_accessor.hpp"
#include "../utils/distance_metrics.hpp"

class HNSWIndex : public ApproximateNN {
private:
    struct HNSWNode {
        uint64_t slot_id;
        std::string key;
        size_t level;
        std::vector<std::vector<size_t>> neighbors;
        std::vector<std::vector<float>> neighbor_dists;

        HNSWNode(uint64_t sid, const std::string& k, size_t lvl);
        void addNeighbor(size_t neighbor_id, float distance, size_t at_level);
        void removeNeighbor(size_t neighbor_id, size_t at_level);
        const std::vector<size_t>& getNeighbors(size_t at_level) const;
        const std::vector<float>& getNeighborDists(size_t at_level) const;
    };

    struct SearchCandidate {
        size_t node_id;
        float distance;
        bool operator<(const SearchCandidate& other) const { return distance < other.distance; }
        bool operator>(const SearchCandidate& other) const { return distance > other.distance; }
    };

    size_t max_connections;
    size_t max_connections_zero;
    size_t ef_construction;
    size_t ef_search;
    float ml;

    std::vector<HNSWNode> nodes;
    std::vector<size_t> entry_points;
    size_t max_level;
    size_t dimensions;
    std::shared_ptr<const DistanceMetric> distance_metric;
    VectorAccessor accessor_;

    std::unordered_set<std::string> deleted_keys_;

    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<float> uniform_dist;

    size_t getRandomLevel() const;
    std::vector<SearchCandidate> searchLayer(const float* query, size_t ef, size_t level) const;
    std::vector<SearchCandidate> searchLayerBase(const float* query, size_t ef) const;
    void addConnections(size_t node_id, const std::vector<SearchCandidate>& candidates, size_t level);
    std::vector<SearchCandidate> selectNeighbors(const std::vector<SearchCandidate>& candidates, size_t M) const;

    float getDistance(const float* a, const float* b) const;

public:
    HNSWIndex(size_t dimensions, size_t M = 16, size_t ef_construction = 200,
              size_t ef_search = 50, std::shared_ptr<const DistanceMetric> metric = nullptr,
              VectorAccessor accessor = nullptr);

    // ApproximateNN interface — insert takes slot_id
    void insert(uint64_t slot_id, const std::string& key) override;
    void remove(const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;

    void setEfSearch(size_t ef);
    size_t getEfSearch() const { return ef_search; }
    size_t getMaxConnections() const { return max_connections; }
    size_t getMaxLevel() const { return max_level; }
    size_t size() const { return nodes.size(); }

    void printStats() const;
    std::vector<size_t> getLevelDistribution() const;
};
