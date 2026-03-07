
#pragma once

#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "approximate_nn.hpp"
#include "../core/vector.hpp"
#include "../utils/distance_metrics.hpp"

class HNSWIndex : public ApproximateNN {
private:
    // HNSW Node structure following hnswlib patterns
    struct HNSWNode {
        Vector vector;
        std::string key;
        size_t level;  // Layer in the hierarchy (0 = bottom layer)
        std::vector<std::vector<size_t>> neighbors;  // neighbors[level] = neighbors at that level
        std::vector<std::vector<float>> neighbor_dists;   // neighbor_dists[level] = distances to neighbors at that level

        HNSWNode(const Vector& vec, const std::string& k, size_t lvl);
        void addNeighbor(size_t neighbor_id, float distance, size_t at_level);
        void removeNeighbor(size_t neighbor_id, size_t at_level);
        const std::vector<size_t>& getNeighbors(size_t at_level) const;
        const std::vector<float>& getNeighborDists(size_t at_level) const;
    };

    // Priority queue element for search
    struct SearchCandidate {
        size_t node_id;
        float distance;
        bool operator<(const SearchCandidate& other) const {
            return distance < other.distance;
        }
        bool operator>(const SearchCandidate& other) const {
            return distance > other.distance;
        }
    };

    // Core HNSW parameters (following hnswlib defaults)
    size_t max_connections;      // M parameter - max connections per layer
    size_t max_connections_zero; // M0 parameter - max connections at layer 0
    size_t ef_construction;      // EF parameter for construction
    size_t ef_search;           // EF parameter for search
    float ml;                   // Maximum layer probability (1/ln(M))
    
    // Data structures
    std::vector<HNSWNode> nodes;
    std::vector<size_t> entry_points;  // Entry points for each level
    size_t max_level;                  // Current maximum level
    size_t dimensions;
    std::shared_ptr<const DistanceMetric> distance_metric;
    
    // Lazy deletion
    std::unordered_set<std::string> deleted_keys_;

    // Random number generation for level assignment
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<float> uniform_dist;

    // Core HNSW algorithms
    size_t getRandomLevel() const;
    std::vector<SearchCandidate> searchLayer(const Vector& query, size_t ef, size_t level) const;
    std::vector<SearchCandidate> searchLayerBase(const Vector& query, size_t ef) const;
    void addConnections(size_t node_id, const std::vector<SearchCandidate>& candidates, size_t level);
    std::vector<SearchCandidate> selectNeighbors(const std::vector<SearchCandidate>& candidates,
                                                 size_t M) const;
    
    // Distance computation helpers
    float getDistance(const Vector& v1, const Vector& v2) const;
    std::vector<float> getDistances(const Vector& query, 
                                   const std::vector<size_t>& node_ids) const;

public:
    // Constructor with hnswlib-style parameters
    HNSWIndex(size_t dimensions, size_t M = 16, size_t ef_construction = 200,
              size_t ef_search = 50, std::shared_ptr<const DistanceMetric> metric = nullptr);
    
    // ApproximateNN interface implementation
    void insert(const Vector& vector, const std::string& key) override;
    void remove(const std::string& key) override;
    std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const override;
    
    // HNSW-specific methods
    void setEfSearch(size_t ef);
    size_t getEfSearch() const { return ef_search; }
    size_t getMaxConnections() const { return max_connections; }
    size_t getMaxLevel() const { return max_level; }
    size_t size() const { return nodes.size(); }
    
    // Statistics and debugging
    void printStats() const;
    std::vector<size_t> getLevelDistribution() const;
};
