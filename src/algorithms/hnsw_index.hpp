
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
        int level;  // Layer in the hierarchy (0 = bottom layer)
        std::vector<std::vector<size_t>> neighbors;  // neighbors[level] = neighbors at that level
        std::vector<std::vector<float>> distances;   // distances[level] = distances to neighbors at that level
        
        HNSWNode(const Vector& vec, const std::string& k, int lvl);
        void addNeighbor(size_t neighbor_id, float distance, int level);
        void removeNeighbor(size_t neighbor_id, int level);
        const std::vector<size_t>& getNeighbors(int level) const;
        const std::vector<float>& getDistances(int level) const;
    };

    // Priority queue element for search
    struct SearchCandidate {
        size_t node_id;
        float distance;
        bool operator>(const SearchCandidate& other) const {
            return distance > other.distance;
        }
    };

    // Visited set element for search
    struct VisitedElement {
        size_t node_id;
        float distance;
        bool operator<(const VisitedElement& other) const {
            return distance < other.distance;
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
    const DistanceMetric* distance_metric;
    
    // Random number generation for level assignment
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<float> uniform_dist;

    // Core HNSW algorithms
    int getRandomLevel() const;
    std::vector<size_t> searchLayer(const Vector& query, size_t ef, int level) const;
    std::vector<size_t> searchLayerBase(const Vector& query, size_t ef) const;
    void addConnections(size_t node_id, const std::vector<size_t>& candidates, int level);
    std::vector<size_t> selectNeighbors(const Vector& query, 
                                       const std::vector<size_t>& candidates, 
                                       size_t M, int level) const;
    std::vector<size_t> selectNeighborsSimple(const std::vector<size_t>& candidates, 
                                             size_t M) const;
    
    // Distance computation helpers
    float getDistance(const Vector& v1, const Vector& v2) const;
    std::vector<float> getDistances(const Vector& query, 
                                   const std::vector<size_t>& node_ids) const;

public:
    // Constructor with hnswlib-style parameters
    HNSWIndex(size_t dimensions, size_t M = 16, size_t ef_construction = 200, 
              size_t ef_search = 50, const DistanceMetric* metric = nullptr);
    
    // ApproximateNN interface implementation
    void insert(const Vector& vector, const std::string& key) override;
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
