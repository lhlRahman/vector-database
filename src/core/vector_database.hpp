#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../algorithms/hnsw_index.hpp"
#include "../core/vector.hpp"
#include "../core/vector_accessor.hpp"
#include "../features/atomic_batch_insert.hpp"
#include "../features/atomic_persistence.hpp"
#include "../features/query_cache.hpp"
#include "../optimizations/rw_lock.hpp"
#include "../optimizations/scalar_quantization.hpp"
#include "../storage/mmap_storage.hpp"
#include "../storage/segmented_vector_store.hpp"

/**
 * Vector Database facade over two storage engines.
 *
 * Default engine: SegmentedVectorStore (WAL-backed mutable segment + sealed HNSW
 * snapshots). Legacy engine: MMapStorage (mmap'd slot file, OS-paged like
 * PostgreSQL). Indexes store slot IDs, not vector copies, reading through a
 * VectorAccessor. Concurrency is a single RWLock (std::shared_mutex): shared for
 * readers, exclusive for writers. (An earlier epoch-RCU was removed as unsound —
 * TSan caught it racing; there is no lock-free read path.)
 */
class VectorDatabase {
    // Sane HNSW defaults (the old 10/8/8 gave ~0.6 recall@10 on the in-memory
    // path). ef_construction=200 builds a good graph; ef_search=64 is a solid
    // recall/latency point (search also clamps ef_search >= k).
    static constexpr size_t kDefaultHNSW_M = 16;
    static constexpr size_t kDefaultHNSW_EfConstruction = 200;
    static constexpr size_t kDefaultHNSW_EfSearch = 64;
    static constexpr size_t kDefaultGPUThreshold = 1000;
    static constexpr size_t kDefaultCacheCapacity = 1000;

public:
    enum class SearchMode {
        Exact,
        HNSW,
    };

    enum class StorageEngine {
        MMap,
        Segmented,
    };

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
        SearchMode search_mode;
        bool atomic_persistence_enabled;
        bool batch_operations_enabled;
        bool query_cache_enabled;
        StorageEngine storage_engine;
        AtomicPersistence::Statistics persistence_stats;
        AtomicBatchInsert::Statistics batch_stats;
        QueryCache::Statistics cache_stats;
        SegmentedVectorStore::Statistics segmented_stats;
    };

private:
    // Core — mmap-backed storage
    std::unique_ptr<MMapStorage> storage_;
    std::unique_ptr<SegmentedVectorStore> segmented_store_;
    std::unordered_map<std::string, uint64_t> key_to_slot_;
    std::string storage_path_;
    StorageEngine storage_engine{StorageEngine::Segmented};

    // Vector accessor — indexes use this to read vector data from mmap
    VectorAccessor vec_accessor_;

    std::shared_ptr<DistanceMetric> distance_metric;
    size_t dimensions;
    mutable RWLock rw_lock_;

    // Approximate index
    SearchMode search_mode{SearchMode::Exact};
    std::unique_ptr<HNSWIndex> hnsw_index;
    size_t hnsw_M{kDefaultHNSW_M};
    size_t hnsw_ef_construction{kDefaultHNSW_EfConstruction};
    size_t hnsw_ef_search{kDefaultHNSW_EfSearch};
    HNSWIndex::AllocationStrategy hnsw_allocation_strategy{HNSWIndex::AllocationStrategy::Standard};
    size_t hnsw_arena_initial_size{1024 * 1024};

    // Features
    bool atomic_persistence_enabled;
    bool batch_operations_enabled;
    bool query_cache_enabled;
    std::shared_ptr<AtomicPersistence> persistence_manager;
    std::unique_ptr<AtomicBatchInsert> batch_manager;
    std::unique_ptr<QueryCache> query_cache;
    PersistenceConfig persistence_config;

    // Scalar quantization for fast candidate filtering
    ScalarQuantizer quantizer_;
    std::vector<uint8_t> quantized_vectors_;  // contiguous N*dims buffer
    std::vector<std::string> quantized_keys_;
    std::vector<uint64_t> quantized_slots_;
    std::atomic<bool> quantizer_dirty_{true};
    static constexpr size_t kQuantizerReRankFactor = 4; // re-rank top k*4

    // State
    std::atomic<bool> ready{false};
    std::atomic<bool> recovering{false};

    // GPU acceleration (protected by gpu_mutex, not rw_lock_)
    bool gpu_enabled{false};
    bool gpu_initialized{false};
    size_t gpu_threshold{kDefaultGPUThreshold};
    std::atomic<bool> gpu_buffer_dirty{true};
    mutable std::mutex gpu_mutex_;

    // Contiguous storage for GPU
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
    void rebuildQuantizer();

public:
    VectorDatabase(size_t dimensions,
                   SearchMode search_mode = SearchMode::Exact,
                   bool enable_atomic_persistence = false,
                   bool enable_batch_operations = false,
                   const PersistenceConfig& persistence_config = {},
                   bool enable_query_cache = true,
                   size_t cache_capacity = kDefaultCacheCapacity,
                   const std::string& storage_path = "",
                   StorageEngine storage_engine = StorageEngine::Segmented);

    ~VectorDatabase() noexcept;

    VectorDatabase(const VectorDatabase&) = delete;
    VectorDatabase& operator=(const VectorDatabase&) = delete;

    void initialize();
    void shutdown();

    void setDistanceMetric(std::shared_ptr<DistanceMetric> metric);

    std::unordered_map<std::string, Vector> getAllVectors() const;

    void setSearchMode(SearchMode mode);
    SearchMode getSearchMode() const;
    void configureHNSW(size_t M, size_t ef_construction, size_t ef_search);
    void configureHNSWAllocator(HNSWIndex::AllocationStrategy strategy,
                                size_t arena_initial_size = 1024 * 1024);
    void configureSegmentedStorage(size_t max_mutable_segment_records,
                                   size_t max_sealed_segments = 16,
                                   double max_tombstone_ratio = 0.25);
    void sealMutableSegment();
    void compactSegments();

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
    std::vector<std::pair<std::string, float>> exactSearch(const Vector& query, size_t k) const;
    std::vector<std::pair<std::string, float>> gpuAcceleratedSearch(const Vector& query, size_t k);
    void rebuildGPUBuffer();
    void markGPUBufferDirty();
    static std::string make_temp_path();
};
