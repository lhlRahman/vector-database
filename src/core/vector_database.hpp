// Copyright [year] <Copyright Owner>
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../algorithms/hnsw_index.hpp"
#include "../algorithms/lsh_index.hpp"
#include "kd_tree.hpp"
#include "../core/vector.hpp"
#include "../features/atomic_batch_insert.hpp"
#include "../features/atomic_persistence.hpp"
#include "../features/dimensionality_reduction.hpp"
#include "../features/query_cache.hpp"
#include "../utils/distance_metrics.hpp"

/**
 * Vector Database with Atomic Persistence
 *
 * Thread-safe, persistent vector DB with exact/approximate search,
 * atomic ops, and batch inserts.
 */
class VectorDatabase {
public:
    struct SearchResult {
        std::string key;
        float distance;
        std::string metadata;
    };

    struct DatabaseStatistics {
        uint64_t total_vectors;
        uint64_t total_inserts;
        uint64_t total_searches;
        uint64_t total_updates;
        uint64_t total_deletes;
        size_t dimensions;
        std::string algorithm;
        bool atomic_persistence_enabled;
        bool batch_operations_enabled;
        AtomicPersistence::Statistics persistence_stats;
        AtomicBatchInsert::Statistics batch_stats;
    };

private:
    // Core
    std::unordered_map<std::string, Vector> vector_map;
    std::unordered_map<std::string, std::string> metadata_map;
    std::unique_ptr<KDTree> kd_tree;
    std::shared_ptr<DistanceMetric> distance_metric;
    size_t dimensions;
    mutable std::mutex db_mutex;   // mutable so const methods can lock

    // Approximate indexes
    std::string approximate_algorithm;
    std::unique_ptr<LSHIndex> lsh_index;
    std::unique_ptr<HNSWIndex> hnsw_index;

    // Features
    bool atomic_persistence_enabled;
    bool batch_operations_enabled;
    std::shared_ptr<AtomicPersistence> persistence_manager;
    std::unique_ptr<AtomicBatchInsert> batch_manager;
    PersistenceConfig persistence_config;

    // State
    std::atomic<bool> ready{false};
    std::atomic<bool> recovering{false};

    // Stats
    std::atomic<uint64_t> total_inserts{0};
    std::atomic<uint64_t> total_searches{0};
    std::atomic<uint64_t> total_updates{0};
    std::atomic<uint64_t> total_deletes{0};
    std::atomic<uint64_t> batch_transaction_counter{0};

    // Private
    void initializeAtomicPersistence();
    void loadExistingData();

public:
    VectorDatabase(size_t dimensions,
                   const std::string& algorithm = "exact",
                   bool enable_atomic_persistence = false,
                   bool enable_batch_operations = false,
                   const PersistenceConfig& persistence_config = {});

    ~VectorDatabase();

    VectorDatabase(const VectorDatabase&) = delete;
    VectorDatabase& operator=(const VectorDatabase&) = delete;

    void initialize();
    void shutdown();

    void setDistanceMetric(std::shared_ptr<DistanceMetric> metric);

    const std::unordered_map<std::string, Vector>& getAllVectors() const;

    void setApproximateAlgorithm(const std::string& algorithm, size_t param1, size_t param2);

    bool insert(const Vector& vector, const std::string& key, const std::string& metadata = "");
    bool insert(const Vector& vector, const std::string& key);

    bool update(const Vector& vector, const std::string& key, const std::string& metadata = "");

    bool remove(const std::string& key);

    const Vector* get(const std::string& key) const;

    std::string getMetadata(const std::string& key) const;

    AtomicBatchInsert::BatchResult batchInsert(const std::vector<std::string>& keys,
                                               const std::vector<Vector>& vectors,
                                               const std::vector<std::string>& metadata = {});
    AtomicBatchInsert::BatchResult batchUpdate(const std::vector<std::string>& keys,
                                               const std::vector<Vector>& vectors,
                                               const std::vector<std::string>& metadata = {});
    AtomicBatchInsert::BatchResult batchDelete(const std::vector<std::string>& keys);

    std::vector<std::pair<std::string, float>> similaritySearch(const Vector& query, size_t k);
    std::vector<std::pair<std::string, float>> similaritySearch(const Vector& query, size_t k) const;

    std::vector<SearchResult> similaritySearchWithMetadata(const Vector& query, size_t k);

    std::vector<std::vector<std::pair<std::string, float>>> batchSimilaritySearch(
        const std::vector<Vector>& queries, size_t k);

    size_t flush();

    bool checkpoint();

    DatabaseStatistics getStatistics() const;

    RecoveryStateMachine::RecoveryInfo getRecoveryInfo() const;

    const PersistenceConfig& getPersistenceConfig() const;

    bool isReady() const;

    void setReady(bool is_ready);

    bool isRecovering() const;

    void setRecovering(bool is_recovering);

    void updatePersistenceConfig(const PersistenceConfig& config);
    
    std::unordered_map<std::string, Vector> getAllVectorsCopy() const;
};
