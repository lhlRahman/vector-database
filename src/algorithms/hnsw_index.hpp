
#pragma once

#include <cstdint>
#include <memory>
#include <memory_resource>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../core/vector.hpp"
#include "../core/vector_accessor.hpp"
#include "../utils/distance_metrics.hpp"
#include "../utils/tracking_memory_resource.hpp"

class HNSWIndex {
public:
    enum class AllocationStrategy {
        Standard,
        Arena,
    };

    struct MemoryStatistics {
        AllocationStrategy strategy;
        size_t allocation_calls;
        size_t deallocation_calls;
        size_t bytes_allocated;
        size_t bytes_deallocated;
        size_t bytes_outstanding;
        size_t peak_bytes_outstanding;
    };

private:
    struct HNSWNode {
        uint64_t slot_id;
        std::string key;
        size_t level;
        std::pmr::vector<std::pmr::vector<size_t>> neighbors;
        std::pmr::vector<std::pmr::vector<float>> neighbor_dists;

        HNSWNode(uint64_t sid, const std::string& k, size_t lvl,
                 size_t max_conn, size_t max_conn_zero, std::pmr::memory_resource* resource);
        void addNeighbor(size_t neighbor_id, float distance, size_t at_level);
        void removeNeighbor(size_t neighbor_id, size_t at_level);
        void assignNeighbors(size_t at_level, const std::vector<size_t>& ids, const std::vector<float>& dists);
        const std::pmr::vector<size_t>& getNeighbors(size_t at_level) const;
        const std::pmr::vector<float>& getNeighborDists(size_t at_level) const;
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

    AllocationStrategy allocation_strategy;
    TrackingMemoryResource tracking_resource_;
    // A pool resource RECYCLES freed blocks, unlike monotonic_buffer_resource
    // which never released memory — so every vector reallocation during graph
    // build accumulated (observed ~2 GB peak for 30k nodes). The pool keeps
    // outstanding memory near the live footprint.
    std::pmr::unsynchronized_pool_resource arena_resource_;
    std::pmr::memory_resource* index_resource_;

    std::pmr::vector<HNSWNode> nodes;
    std::pmr::vector<size_t> entry_points;
    size_t max_level;
    size_t dimensions;
    std::shared_ptr<const DistanceMetric> distance_metric;
    // The concrete metric resolved once at construction, so getDistance() on the
    // hot graph-walk avoids a virtual call and can inline the SIMD kernel.
    enum class MetricKind { Euclidean, Manhattan, Cosine, Virtual };
    MetricKind metric_kind_{MetricKind::Virtual};
    VectorAccessor accessor_;

    std::unordered_set<std::string> deleted_keys_;
    std::unordered_set<uint64_t> deleted_slots_;
    // key -> node index of the current live node for that key, and nodes
    // tombstoned by a re-insert of the same key (MMap in-place update reuses the
    // slot, so slot-based filtering alone can't distinguish old from new).
    // Runtime-only; the segmented path also records deleted_slots_ (serialized).
    std::unordered_map<std::string, size_t> key_to_node_;
    std::unordered_set<size_t> deleted_node_ids_;

    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<float> uniform_dist;

    size_t getRandomLevel() const;
    // Greedy best-first search of a single layer starting from an explicit entry
    // point (the closest node found while descending the layer above).
    std::vector<SearchCandidate> searchLayer(const float* query, size_t ef, size_t level, size_t entry_point) const;
    void addConnections(size_t node_id, const std::vector<SearchCandidate>& candidates, size_t level);
    // Malkov-Yashunin heuristic neighbor selection (Alg. 4): keeps diverse /
    // long-range edges instead of just the M closest. `candidates[i].distance`
    // is the distance to the reference point (query on insert, the node itself
    // when re-pruning a neighbor list).
    std::vector<SearchCandidate> selectNeighbors(const std::vector<SearchCandidate>& candidates, size_t M) const;
    // Re-select a node's level-`level` neighbor list down to M using the heuristic.
    void pruneNeighbors(size_t node_id, size_t level, size_t M);

    float getDistance(const float* a, const float* b) const;

public:
    struct NodeSnapshot {
        uint64_t slot_id;
        std::string key;
        size_t level;
        std::vector<std::vector<size_t>> neighbors;
        std::vector<std::vector<float>> neighbor_dists;
    };

    struct GraphSnapshot {
        size_t dimensions;
        size_t max_connections;
        size_t max_connections_zero;
        size_t ef_construction;
        size_t ef_search;
        size_t max_level;
        std::vector<size_t> entry_points;
        std::vector<NodeSnapshot> nodes;
        std::vector<std::string> deleted_keys;
        std::vector<uint64_t> deleted_slots;
    };

    HNSWIndex(size_t dimensions, size_t M = 16, size_t ef_construction = 200,
              size_t ef_search = 50, std::shared_ptr<const DistanceMetric> metric = nullptr,
              VectorAccessor accessor = nullptr,
              AllocationStrategy allocation_strategy = AllocationStrategy::Standard,
              size_t arena_initial_size = 1024 * 1024);

    void insert(uint64_t slot_id, const std::string& key);
    void remove(const std::string& key);
    void removeSlot(uint64_t slot_id);
    [[nodiscard]] std::vector<std::pair<std::string, float>> search(const Vector& query, size_t k) const;

    void setEfSearch(size_t ef);
    size_t getEfSearch() const { return ef_search; }
    size_t getMaxConnections() const { return max_connections; }
    size_t getMaxLevel() const { return max_level; }
    size_t size() const { return nodes.size(); }
    MemoryStatistics getMemoryStatistics() const;
    GraphSnapshot exportGraph() const;
    void importGraph(const GraphSnapshot& snapshot);

    void printStats() const;
    std::vector<size_t> getLevelDistribution() const;
};
