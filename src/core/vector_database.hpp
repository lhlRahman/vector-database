#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../algorithms/hnsw_index.hpp"
#include "../algorithms/lsh_index.hpp"
#include "kd_tree.hpp"
#include "../core/vector.hpp"
#include "../features/atomic_batch_insert.hpp"
#include "../features/atomic_persistence.hpp"
#include "../features/query_cache.hpp"
#include "../utils/distance_metrics.hpp"

/**
 * Vector Database with Atomic Persistence
 *
 * Thread-safe, persistent vector DB with exact/approximate search,
 * atomic ops, and batch inserts.
 */
class VectorDatabase {
    static constexpr size_t kDefaultLSHTables = 10;
    static constexpr size_t kDefaultLSHHashFunctions = 8;
    static constexpr size_t kDefaultHNSW_M = 10;
    static constexpr size_t kDefaultHNSW_EfConstruction = 8;
    static constexpr size_t kDefaultHNSW_EfSearch = 8;
    static constexpr size_t kDefaultGPUThreshold = 1000;
    static constexpr size_t kDefaultCacheCapacity = 1000;

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
        bool query_cache_enabled;
        AtomicPersistence::Statistics persistence_stats;
        AtomicBatchInsert::Statistics batch_stats;
        QueryCache::Statistics cache_stats;
    };

private:
    // Core
    std::unordered_map<std::string, Vector> vector_map;
    std::unordered_map<std::string, std::string> metadata_map;
    std::unique_ptr<KDTree> kd_tree;
    std::shared_ptr<DistanceMetric> distance_metric;
    size_t dimensions;
    mutable std::shared_mutex db_mutex;  // shared for reads, exclusive for writes

    // Approximate indexes
    std::string approximate_algorithm;
    std::unique_ptr<LSHIndex> lsh_index;
    std::unique_ptr<HNSWIndex> hnsw_index;
    size_t lsh_num_tables{kDefaultLSHTables};
    size_t lsh_num_hash_functions{kDefaultLSHHashFunctions};
    size_t hnsw_M{kDefaultHNSW_M};
    size_t hnsw_ef_construction{kDefaultHNSW_EfConstruction};
    size_t hnsw_ef_search{kDefaultHNSW_EfSearch};

    // Features
    bool atomic_persistence_enabled;
    bool batch_operations_enabled;
    bool query_cache_enabled;
    std::shared_ptr<AtomicPersistence> persistence_manager;
    std::unique_ptr<AtomicBatchInsert> batch_manager;
    std::unique_ptr<QueryCache> query_cache;
    PersistenceConfig persistence_config;

    // State
    std::atomic<bool> ready{false};
    std::atomic<bool> recovering{false};
    
    // GPU acceleration (protected by gpu_mutex, not db_mutex)
    bool gpu_enabled{false};
    bool gpu_initialized{false};
    size_t gpu_threshold{kDefaultGPUThreshold};
    std::atomic<bool> gpu_buffer_dirty{true};
    mutable std::mutex gpu_mutex_;

    // Contiguous storage for GPU (mirrors vector_map, protected by gpu_mutex_)
    std::vector<float> flat_vectors;
    std::vector<std::string> vector_keys;

    // Stats
    std::atomic<uint64_t> total_inserts{0};
    std::atomic<uint64_t> total_searches{0};
    std::atomic<uint64_t> total_updates{0};
    std::atomic<uint64_t> total_deletes{0};
    std::atomic<uint64_t> batch_transaction_counter{0};

    // Private
    void initializeAtomicPersistence();
    void loadExistingData();
    void rebuildIndexes();

public:
    VectorDatabase(size_t dimensions,
                   const std::string& algorithm = "exact",
                   bool enable_atomic_persistence = false,
                   bool enable_batch_operations = false,
                   const PersistenceConfig& persistence_config = {},
                   bool enable_query_cache = true,
                   size_t cache_capacity = kDefaultCacheCapacity);

    ~VectorDatabase() noexcept;

    VectorDatabase(const VectorDatabase&) = delete;
    VectorDatabase& operator=(const VectorDatabase&) = delete;

    void initialize();
    void shutdown();

    void setDistanceMetric(std::shared_ptr<DistanceMetric> metric);

    std::unordered_map<std::string, Vector> getAllVectors() const;

    void setApproximateAlgorithm(const std::string& algorithm, size_t param1, size_t param2);

    [[nodiscard]] bool insert(const Vector& vector, const std::string& key, const std::string& metadata = "");

    [[nodiscard]] bool update(const Vector& vector, const std::string& key, const std::string& metadata = "");

    [[nodiscard]] bool remove(const std::string& key);

    [[nodiscard]] std::optional<Vector> get(const std::string& key) const;

    std::string getMetadata(const std::string& key) const;

    AtomicBatchInsert::BatchResult batchInsert(const std::vector<std::string>& keys,
                                               const std::vector<Vector>& vectors,
                                               const std::vector<std::string>& metadata = {});
    AtomicBatchInsert::BatchResult batchUpdate(const std::vector<std::string>& keys,
                                               const std::vector<Vector>& vectors,
                                               const std::vector<std::string>& metadata = {});
    AtomicBatchInsert::BatchResult batchDelete(const std::vector<std::string>& keys);

    std::vector<std::pair<std::string, float>> similaritySearch(const Vector& query, size_t k);

    std::vector<SearchResult> similaritySearchWithMetadata(const Vector& query, size_t k);

    std::vector<std::vector<std::pair<std::string, float>>> batchSimilaritySearch(
        const std::vector<Vector>& queries, size_t k);

    [[nodiscard]] size_t flush();

    [[nodiscard]] bool checkpoint();

    DatabaseStatistics getStatistics() const;

    const PersistenceConfig& getPersistenceConfig() const;

    bool isReady() const;

    void setReady(bool is_ready);

    bool isRecovering() const;

    void setRecovering(bool is_recovering);

    void updatePersistenceConfig(const PersistenceConfig& config);
    
    size_t vectorCount() const;
    
    // SIMD control
    void enableSIMD(bool enable);
    bool isSIMDEnabled() const;
    
    // GPU acceleration control
    void enableGPU(bool enable);
    bool isGPUEnabled() const;
    bool isGPUAvailable() const;
    void setGPUThreshold(size_t threshold);
    size_t getGPUThreshold() const;
    
private:
    // GPU-accelerated search (called internally when conditions are met)
    std::vector<std::pair<std::string, float>> gpuAcceleratedSearch(const Vector& query, size_t k);
    
    // Rebuild contiguous storage and GPU buffer
    void rebuildGPUBuffer();
    void markGPUBufferDirty();
};
