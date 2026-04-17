#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vector_database.hpp"
#include "../algorithms/flat_index.hpp"
#include "../optimizations/gpu_operations.hpp"
#include "../optimizations/simd_operations.hpp"

// -------------------- ctor / dtor --------------------

std::string VectorDatabase::make_temp_path() {
    auto tmp = std::filesystem::temp_directory_path();
    auto name = "vdb_" + std::to_string(getpid()) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".vdb";
    return (tmp / name).string();
}

VectorDatabase::VectorDatabase(size_t dimensions,
                               SearchMode mode,
                               bool enable_atomic_persistence,
                               bool enable_batch_operations,
                               const PersistenceConfig& persistence_config,
                               bool enable_query_cache,
                               size_t cache_capacity,
                               const std::string& storage_path,
                               StorageEngine engine)
    : storage_path_(storage_path.empty() ? make_temp_path() : storage_path),
      storage_engine(engine),
      dimensions(dimensions),
      search_mode(mode),
      atomic_persistence_enabled(enable_atomic_persistence),
      batch_operations_enabled(enable_batch_operations),
      query_cache_enabled(enable_query_cache),
      persistence_config(persistence_config),
      quantizer_(dimensions),
      total_inserts(0),
      total_searches(0),
      total_updates(0),
      total_deletes(0) {
    // Distance metric
    distance_metric = std::make_shared<EuclideanDistance>();

    if (storage_engine == StorageEngine::Segmented) {
        SegmentedVectorStore::Config config;
        config.dimensions = dimensions;
        config.hnsw_m = hnsw_M;
        config.hnsw_ef_construction = hnsw_ef_construction;
        config.hnsw_ef_search = hnsw_ef_search;
        config.allocation_strategy = hnsw_allocation_strategy;
        config.arena_initial_size = hnsw_arena_initial_size;
        config.metric = distance_metric;
        segmented_store_ = std::make_unique<SegmentedVectorStore>(storage_path_, std::move(config));
    } else {
        // Strict order: storage_ must exist before vec_accessor_ captures it,
        // and vec_accessor_ must be bound before any index that uses it.
        // The previous code initialized the accessor up-front with a `[this]`
        // lambda that resolved storage_ lazily — it worked, but the ordering
        // requirement was invisible to a reader and easy to break by accident.
        storage_ = std::make_unique<MMapStorage>(storage_path_, dimensions);
        vec_accessor_ = [storage = storage_.get()](uint64_t slot_id) -> const float* {
            return storage->vector_ptr(slot_id);
        };

        if (search_mode == SearchMode::HNSW) {
            hnsw_M = kDefaultHNSW_M;
            hnsw_ef_construction = kDefaultHNSW_EfConstruction;
            hnsw_ef_search = kDefaultHNSW_EfSearch;
            hnsw_index = std::make_unique<HNSWIndex>(
                dimensions, hnsw_M, hnsw_ef_construction, hnsw_ef_search,
                distance_metric, vec_accessor_, hnsw_allocation_strategy, hnsw_arena_initial_size);
        }
    }

    // Query cache
    if (enable_query_cache) {
        query_cache = std::make_unique<QueryCache>(cache_capacity);
    }
}

VectorDatabase::~VectorDatabase() noexcept {
    try {
        shutdown();
    } catch (...) {}
    if (storage_engine == StorageEngine::MMap &&
        storage_path_.find("vdb_") != std::string::npos &&
        storage_path_.find(std::filesystem::temp_directory_path().string()) != std::string::npos) {
        std::filesystem::remove(storage_path_);
    }
}

// -------------------- lifecycle --------------------

void VectorDatabase::initialize() {
    RWLock::WriteGuard wg(rw_lock_);

    if (ready.load()) return;

    if (storage_engine == StorageEngine::Segmented) {
        segmented_store_->initialize();
        ready.store(true);
        return;
    }

    // Open mmap storage
    storage_->open();

    // Use sequential access pattern for bulk loading
    storage_->advise_sequential();

    // Rebuild key→slot index from mmap'd data
    key_to_slot_ = storage_->build_key_index();

    // Rebuild in-memory index from mmap'd vectors (using slot IDs, zero-copy)
    for (const auto& [key, slot_id] : key_to_slot_) {
        if (hnsw_index) hnsw_index->insert(slot_id, key);
    }

    // Switch to random access for normal operation
    storage_->advise_random();

    if (atomic_persistence_enabled) {
        initializeAtomicPersistence();
    }

    // Build scalar quantization index
    quantizer_dirty_.store(true);
    if (key_to_slot_.size() > 0) {
        rebuildQuantizer();
    }

    ready.store(true);
}

void VectorDatabase::shutdown() {
    RWLock::WriteGuard wg(rw_lock_);

    if (!ready.load()) return;

    if (segmented_store_) {
        segmented_store_->shutdown();
    }

    if (persistence_manager) {
        persistence_manager->shutdown();
    }

    if (storage_) {
        storage_->sync();
        storage_->close();
    }

    ready.store(false);
}

void VectorDatabase::initializeAtomicPersistence() {
    persistence_manager = std::make_shared<AtomicPersistence>(persistence_config);
    persistence_manager->initialize();

    if (batch_operations_enabled) {
        batch_manager = std::make_unique<AtomicBatchInsert>(persistence_manager);
    }
}

void VectorDatabase::loadExistingData() {
    if (!persistence_manager) return;

    std::unordered_map<std::string, Vector> loaded_vectors;
    std::unordered_map<std::string, std::string> loaded_metadata;

    if (persistence_manager->loadDatabase(loaded_vectors, loaded_metadata)) {
        for (const auto& [key, vector] : loaded_vectors) {
            uint64_t slot = storage_->insert(key, vector.data_ptr(), "");
            key_to_slot_[key] = slot;

            if (hnsw_index) hnsw_index->insert(slot, key);
        }

    }
}

// -------------------- configuration --------------------

void VectorDatabase::setDistanceMetric(std::shared_ptr<DistanceMetric> metric) {
    RWLock::WriteGuard wg(rw_lock_);

    distance_metric = std::move(metric);
    if (segmented_store_) {
        segmented_store_->setMetric(distance_metric);
    }
    rebuildIndexes();

    if (query_cache) {
        query_cache->clear();
    }
}

void VectorDatabase::setSearchMode(SearchMode mode) {
    RWLock::WriteGuard wg(rw_lock_);

    search_mode = mode;
    if (storage_engine == StorageEngine::MMap) {
        rebuildIndexes();
    }

    if (query_cache) {
        query_cache->clear();
    }
}

VectorDatabase::SearchMode VectorDatabase::getSearchMode() const {
    RWLock::ReadGuard rg(rw_lock_);
    return search_mode;
}

void VectorDatabase::configureHNSW(size_t M, size_t ef_construction, size_t ef_search) {
    RWLock::WriteGuard wg(rw_lock_);

    hnsw_M = M;
    hnsw_ef_construction = ef_construction;
    hnsw_ef_search = ef_search;
    if (segmented_store_) {
        segmented_store_->configureHNSW(M, ef_construction, ef_search);
    }
    if (storage_engine == StorageEngine::MMap) {
        rebuildIndexes();
    }

    if (query_cache) {
        query_cache->clear();
    }
}

void VectorDatabase::configureHNSWAllocator(HNSWIndex::AllocationStrategy strategy, size_t arena_initial_size) {
    RWLock::WriteGuard wg(rw_lock_);

    hnsw_allocation_strategy = strategy;
    hnsw_arena_initial_size = arena_initial_size;
    if (segmented_store_) {
        segmented_store_->configureAllocator(strategy, arena_initial_size);
    }
    if (storage_engine == StorageEngine::MMap) {
        rebuildIndexes();
    }

    if (query_cache) {
        query_cache->clear();
    }
}

void VectorDatabase::configureSegmentedStorage(size_t max_mutable_segment_records,
                                               size_t max_sealed_segments,
                                               double max_tombstone_ratio) {
    RWLock::WriteGuard wg(rw_lock_);
    if (segmented_store_) {
        segmented_store_->configureSegmentation(max_mutable_segment_records,
                                                max_sealed_segments,
                                                max_tombstone_ratio);
    }
}

void VectorDatabase::sealMutableSegment() {
    RWLock::WriteGuard wg(rw_lock_);
    if (segmented_store_) {
        segmented_store_->sealMutableSegment();
        if (query_cache) query_cache->clear();
    }
}

void VectorDatabase::compactSegments() {
    RWLock::WriteGuard wg(rw_lock_);
    if (segmented_store_) {
        segmented_store_->compact();
        if (query_cache) query_cache->clear();
    }
}

void VectorDatabase::rebuildIndexes() {
    if (storage_engine == StorageEngine::Segmented) return;

    // Use sequential access for bulk rebuild
    storage_->advise_sequential();

    hnsw_index.reset();

    if (search_mode == SearchMode::HNSW) {
        hnsw_index = std::make_unique<HNSWIndex>(
            dimensions, hnsw_M, hnsw_ef_construction, hnsw_ef_search,
            distance_metric, vec_accessor_, hnsw_allocation_strategy, hnsw_arena_initial_size);
    }

    for (const auto& [key, slot_id] : key_to_slot_) {
        if (hnsw_index) hnsw_index->insert(slot_id, key);
    }

    // Back to random access
    storage_->advise_random();
}

void VectorDatabase::rebuildQuantizer() {
    size_t n = key_to_slot_.size();
    if (n == 0) {
        quantizer_dirty_.store(false);
        return;
    }

    // Collect all vector pointers for training
    std::vector<const float*> ptrs;
    ptrs.reserve(n);
    quantized_keys_.clear();
    quantized_keys_.reserve(n);
    quantized_slots_.clear();
    quantized_slots_.reserve(n);

    for (const auto& [key, slot_id] : key_to_slot_) {
        ptrs.push_back(storage_->vector_ptr(slot_id));
        quantized_keys_.push_back(key);
        quantized_slots_.push_back(slot_id);
    }

    // Train quantizer on all vectors
    quantizer_.train(ptrs.data(), ptrs.size());

    // Quantize all vectors
    quantized_vectors_.resize(n * dimensions);
    quantizer_.quantize_batch(ptrs.data(), quantized_vectors_.data(), ptrs.size());

    quantizer_dirty_.store(false);
}

// -------------------- mutations --------------------

bool VectorDatabase::insert(const Vector& vector, const std::string& key, const std::string& metadata) {
    RWLock::WriteGuard wg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    if (std::ranges::any_of(vector, [](float f) { return std::isnan(f); })) {
        std::cerr << "Warning: Vector " << key << " contains NaN values. Skipping insertion.\n";
        return false;
    }

    if (key_to_slot_.count(key)) return false;

    if (storage_engine == StorageEngine::Segmented) {
        bool inserted = segmented_store_->insert(vector, key, metadata);
        if (inserted) {
            if (query_cache) query_cache->invalidate();
            total_inserts.fetch_add(1, std::memory_order_relaxed);
        }
        return inserted;
    }

    uint64_t slot_id = storage_->insert(key, vector.data_ptr(), metadata);
    key_to_slot_[key] = slot_id;

    try {
        if (hnsw_index) hnsw_index->insert(slot_id, key);
    } catch (...) {
        storage_->remove(slot_id);
        key_to_slot_.erase(key);
        throw;
    }

    if (query_cache) query_cache->invalidate();
    markGPUBufferDirty();
    quantizer_dirty_.store(true);

    if (persistence_manager) {
        if (!persistence_manager->insert(key, vector, metadata)) {
            storage_->remove(slot_id);
            key_to_slot_.erase(key);
            if (hnsw_index) hnsw_index->remove(key);
            return false;
        }
    }

    total_inserts.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool VectorDatabase::update(const Vector& vector, const std::string& key, const std::string& metadata) {
    RWLock::WriteGuard wg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    if (storage_engine == StorageEngine::Segmented) {
        bool updated = segmented_store_->update(vector, key, metadata);
        if (updated) {
            if (query_cache) query_cache->invalidate();
            total_updates.fetch_add(1, std::memory_order_relaxed);
        }
        return updated;
    }

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return false;

    uint64_t slot_id = it->second;

    const float* old_ptr = storage_->vector_ptr(slot_id);
    std::vector<float> old_data(old_ptr, old_ptr + dimensions);
    std::string old_metadata = storage_->get_metadata(slot_id);

    storage_->update(slot_id, vector.data_ptr(), metadata);

    try {
        if (hnsw_index) { hnsw_index->remove(key); hnsw_index->insert(slot_id, key); }
    } catch (...) {
        storage_->update(slot_id, old_data.data(), old_metadata);
        throw;
    }

    if (query_cache) query_cache->invalidate();
    markGPUBufferDirty();
    quantizer_dirty_.store(true);

    if (persistence_manager) {
        if (!persistence_manager->update(key, vector, metadata)) {
            storage_->update(slot_id, old_data.data(), old_metadata);
            if (hnsw_index) { hnsw_index->remove(key); hnsw_index->insert(slot_id, key); }
            return false;
        }
    }

    total_updates.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool VectorDatabase::remove(const std::string& key) {
    RWLock::WriteGuard wg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");

    if (storage_engine == StorageEngine::Segmented) {
        bool removed = segmented_store_->remove(key);
        if (removed) {
            if (query_cache) query_cache->invalidate();
            total_deletes.fetch_add(1, std::memory_order_relaxed);
        }
        return removed;
    }

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return false;

    uint64_t slot_id = it->second;

    const float* old_ptr = storage_->vector_ptr(slot_id);
    std::vector<float> old_data(old_ptr, old_ptr + dimensions);
    std::string old_metadata = storage_->get_metadata(slot_id);

    storage_->remove(slot_id);
    key_to_slot_.erase(it);

    if (hnsw_index) hnsw_index->remove(key);

    if (query_cache) query_cache->invalidate();
    markGPUBufferDirty();
    quantizer_dirty_.store(true);

    if (persistence_manager) {
        if (!persistence_manager->remove(key)) {
            uint64_t new_slot = storage_->insert(key, old_data.data(), old_metadata);
            key_to_slot_[key] = new_slot;
            if (hnsw_index) hnsw_index->insert(new_slot, key);
            return false;
        }
    }

    total_deletes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// -------------------- queries --------------------

std::vector<std::pair<std::string, float>>
VectorDatabase::exactSearch(const Vector& query, size_t k) const {
    if (!distance_metric) {
        throw std::runtime_error("Distance metric not configured");
    }

    if (dynamic_cast<const EuclideanDistance*>(distance_metric.get()) != nullptr) {
        return FlatIndex<EuclideanMetricPolicy>(dimensions, vec_accessor_).search(query, k, key_to_slot_);
    }
    if (dynamic_cast<const ManhattanDistance*>(distance_metric.get()) != nullptr) {
        return FlatIndex<ManhattanMetricPolicy>(dimensions, vec_accessor_).search(query, k, key_to_slot_);
    }
    if (dynamic_cast<const CosineSimilarity*>(distance_metric.get()) != nullptr) {
        return FlatIndex<CosineMetricPolicy>(dimensions, vec_accessor_).search(query, k, key_to_slot_);
    }

    return FlatIndex<VirtualMetricPolicy>(
        dimensions,
        vec_accessor_,
        VirtualMetricPolicy(distance_metric)).search(query, k, key_to_slot_);
}

std::optional<Vector> VectorDatabase::get(const std::string& key) const {
    RWLock::ReadGuard rg(rw_lock_);

    if (storage_engine == StorageEngine::Segmented) {
        return segmented_store_->get(key);
    }

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return std::nullopt;

    const float* ptr = storage_->vector_ptr(it->second);
    return Vector(std::vector<float>(ptr, ptr + dimensions));
}

std::string VectorDatabase::getMetadata(const std::string& key) const {
    RWLock::ReadGuard rg(rw_lock_);

    if (storage_engine == StorageEngine::Segmented) {
        return segmented_store_->getMetadata(key);
    }

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return "";

    return storage_->get_metadata(it->second);
}

std::vector<std::pair<std::string, float>> VectorDatabase::similaritySearch(const Vector& query, size_t k) {
    RWLock::ReadGuard rg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");

    if (storage_engine == StorageEngine::Segmented) {
        total_searches.fetch_add(1, std::memory_order_relaxed);
        std::vector<std::pair<std::string, float>> results;
        if (query_cache && query_cache->get(query, results)) return results;
        results = segmented_store_->search(query, k);
        if (query_cache) query_cache->put(query, results);
        return results;
    }

    if (key_to_slot_.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    std::vector<std::pair<std::string, float>> results;
    if (query_cache && query_cache->get(query, results)) {
        return results;
    }

    if (gpu_enabled && key_to_slot_.size() > gpu_threshold) {
        results = gpuAcceleratedSearch(query, k);
    }
    else if (search_mode == SearchMode::HNSW && hnsw_index) {
        results = hnsw_index->search(query, k);
    } else {
        results = exactSearch(query, k);
    }

    if (query_cache) {
        query_cache->put(query, results);
    }

    return results;
}

std::vector<VectorDatabase::SearchResult>
VectorDatabase::similaritySearchWithMetadata(const Vector& query, size_t k) {
    RWLock::ReadGuard rg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");

    if (storage_engine == StorageEngine::Segmented) {
        total_searches.fetch_add(1, std::memory_order_relaxed);
        auto raw = segmented_store_->searchWithMetadata(query, k);
        std::vector<SearchResult> results;
        results.reserve(raw.size());
        for (const auto& result : raw) {
            results.push_back(SearchResult{result.key, result.distance, result.metadata});
        }
        return results;
    }

    if (key_to_slot_.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    std::vector<std::pair<std::string, float>> rawResults;
    if (query_cache && query_cache->get(query, rawResults)) {
        // cache hit
    } else if (gpu_enabled && key_to_slot_.size() > gpu_threshold) {
        rawResults = gpuAcceleratedSearch(query, k);
    } else if (search_mode == SearchMode::HNSW && hnsw_index) {
        rawResults = hnsw_index->search(query, k);
    } else {
        rawResults = exactSearch(query, k);
    }
    if (query_cache) {
        query_cache->put(query, rawResults);
    }

    std::vector<SearchResult> results;
    results.reserve(rawResults.size());
    for (const auto& [key, distance] : rawResults) {
        auto it = key_to_slot_.find(key);
        std::string meta = (it != key_to_slot_.end()) ? storage_->get_metadata(it->second) : "";
        results.emplace_back(SearchResult{key, distance, meta});
    }

    return results;
}

std::vector<std::vector<std::pair<std::string, float>>>
VectorDatabase::batchSimilaritySearch(const std::vector<Vector>& queries, size_t k) {
    RWLock::ReadGuard rg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");

    std::vector<std::vector<std::pair<std::string, float>>> results;
    results.reserve(queries.size());

    for (const auto& query : queries) {
        if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
        if (storage_engine == StorageEngine::Segmented) {
            total_searches.fetch_add(1, std::memory_order_relaxed);
            results.push_back(segmented_store_->search(query, k));
            continue;
        }
        if (key_to_slot_.empty()) { results.emplace_back(); continue; }

        total_searches.fetch_add(1, std::memory_order_relaxed);

        std::vector<std::pair<std::string, float>> single_result;
        if (query_cache && query_cache->get(query, single_result)) {
            results.push_back(std::move(single_result));
            continue;
        }

        if (gpu_enabled && key_to_slot_.size() > gpu_threshold) {
            single_result = gpuAcceleratedSearch(query, k);
        } else if (search_mode == SearchMode::HNSW && hnsw_index) {
            single_result = hnsw_index->search(query, k);
        } else {
            single_result = exactSearch(query, k);
        }

        if (query_cache) {
            query_cache->put(query, single_result);
        }

        results.push_back(std::move(single_result));
    }

    return results;
}

// -------------------- batch --------------------

AtomicBatchInsert::BatchResult VectorDatabase::batchInsert(const std::vector<std::string>& keys,
                                                           const std::vector<Vector>& vectors,
                                                           const std::vector<std::string>& metadata) {
    if (!batch_operations_enabled) {
        throw std::runtime_error("Batch operations not enabled");
    }

    if (keys.size() != vectors.size()) {
        return AtomicBatchInsert::BatchResult{false, 0, "Keys and vectors size mismatch", 0, std::chrono::duration<double>(0)};
    }

    auto start_time = std::chrono::steady_clock::now();
    AtomicBatchInsert::BatchResult result;
    result.transaction_id = ++batch_transaction_counter;
    result.success = true;
    result.operations_committed = 0;

    {
        RWLock::WriteGuard wg(rw_lock_);

        if (storage_engine == StorageEngine::Segmented) {
            for (size_t i = 0; i < keys.size(); ++i) {
                const std::string& meta = (i < metadata.size()) ? metadata[i] : "";
                if (segmented_store_->insert(vectors[i], keys[i], meta)) {
                    result.operations_committed++;
                }
            }
            if (query_cache) query_cache->invalidate();
            total_inserts.fetch_add(result.operations_committed, std::memory_order_relaxed);
            result.duration = std::chrono::steady_clock::now() - start_time;
            return result;
        }

        // Use sequential access for bulk insert
        storage_->advise_sequential();

        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string& key = keys[i];
            const Vector& vector = vectors[i];
            const std::string& meta = (i < metadata.size()) ? metadata[i] : "";

            if (key_to_slot_.count(key)) continue;

            if (vector.size() != dimensions) {
                result.success = false;
                result.error_message = "Vector dimension mismatch for key: " + key;
                break;
            }

            uint64_t slot_id = storage_->insert(key, vector.data_ptr(), meta);
            key_to_slot_[key] = slot_id;

            if (hnsw_index) hnsw_index->insert(slot_id, key);

            if (persistence_manager) {
                if (!persistence_manager->insert(key, vector, meta)) {
                    storage_->remove(slot_id);
                    key_to_slot_.erase(key);
                    result.success = false;
                    result.error_message = "Failed to persist key: " + key;
                    break;
                }
            }

            result.operations_committed++;
        }

        // Back to random access
        storage_->advise_random();

        if (query_cache) query_cache->invalidate();
        markGPUBufferDirty();
        quantizer_dirty_.store(true);
    }

    if (result.success) {
        total_inserts.fetch_add(result.operations_committed, std::memory_order_relaxed);
    }

    result.duration = std::chrono::steady_clock::now() - start_time;
    return result;
}

AtomicBatchInsert::BatchResult VectorDatabase::batchUpdate(const std::vector<std::string>& keys,
                                                           const std::vector<Vector>& vectors,
                                                           const std::vector<std::string>& metadata) {
    if (!batch_operations_enabled) {
        throw std::runtime_error("Batch operations not enabled");
    }

    if (keys.size() != vectors.size()) {
        return AtomicBatchInsert::BatchResult{false, 0, "Keys and vectors size mismatch", 0, std::chrono::duration<double>(0)};
    }

    auto start_time = std::chrono::steady_clock::now();
    AtomicBatchInsert::BatchResult result;
    result.transaction_id = ++batch_transaction_counter;
    result.success = true;
    result.operations_committed = 0;

    {
        RWLock::WriteGuard wg(rw_lock_);

        if (storage_engine == StorageEngine::Segmented) {
            for (size_t i = 0; i < keys.size(); ++i) {
                const std::string& meta = (i < metadata.size()) ? metadata[i] : "";
                if (segmented_store_->update(vectors[i], keys[i], meta)) {
                    result.operations_committed++;
                }
            }
            if (query_cache) query_cache->invalidate();
            total_updates.fetch_add(result.operations_committed, std::memory_order_relaxed);
            result.duration = std::chrono::steady_clock::now() - start_time;
            return result;
        }

        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string& key = keys[i];
            const Vector& vector = vectors[i];
            const std::string& meta = (i < metadata.size()) ? metadata[i] : "";

            auto it = key_to_slot_.find(key);
            if (it == key_to_slot_.end()) continue;

            if (vector.size() != dimensions) {
                result.success = false;
                result.error_message = "Vector dimension mismatch for key: " + key;
                break;
            }

            storage_->update(it->second, vector.data_ptr(), meta);

            if (hnsw_index) { hnsw_index->remove(key); hnsw_index->insert(it->second, key); }

            if (persistence_manager) {
                if (!persistence_manager->update(key, vector, meta)) {
                    result.success = false;
                    result.error_message = "Failed to persist update for key: " + key;
                    break;
                }
            }

            result.operations_committed++;
        }

        if (query_cache) query_cache->invalidate();
        markGPUBufferDirty();
        quantizer_dirty_.store(true);
    }

    if (result.success) {
        total_updates.fetch_add(result.operations_committed, std::memory_order_relaxed);
    }

    result.duration = std::chrono::steady_clock::now() - start_time;
    return result;
}

AtomicBatchInsert::BatchResult VectorDatabase::batchDelete(const std::vector<std::string>& keys) {
    if (!batch_operations_enabled) {
        throw std::runtime_error("Batch operations not enabled");
    }

    auto start_time = std::chrono::steady_clock::now();
    AtomicBatchInsert::BatchResult result;
    result.transaction_id = ++batch_transaction_counter;
    result.success = true;
    result.operations_committed = 0;

    {
        RWLock::WriteGuard wg(rw_lock_);

        if (storage_engine == StorageEngine::Segmented) {
            for (const std::string& key : keys) {
                if (segmented_store_->remove(key)) {
                    result.operations_committed++;
                }
            }
            if (query_cache) query_cache->invalidate();
            total_deletes.fetch_add(result.operations_committed, std::memory_order_relaxed);
            result.duration = std::chrono::steady_clock::now() - start_time;
            return result;
        }

        for (const std::string& key : keys) {
            auto it = key_to_slot_.find(key);
            if (it == key_to_slot_.end()) continue;

            storage_->remove(it->second);
            key_to_slot_.erase(it);

            if (hnsw_index) hnsw_index->remove(key);

            if (persistence_manager) {
                if (!persistence_manager->remove(key)) {
                    result.success = false;
                    result.error_message = "Failed to persist deletion for key: " + key;
                    break;
                }
            }

            result.operations_committed++;
        }

        if (query_cache) query_cache->invalidate();
        markGPUBufferDirty();
        quantizer_dirty_.store(true);
    }

    if (result.success) {
        total_deletes.fetch_add(result.operations_committed, std::memory_order_relaxed);
    }

    result.duration = std::chrono::steady_clock::now() - start_time;
    return result;
}

// -------------------- maintenance / stats --------------------

size_t VectorDatabase::flush() {
    RWLock::ReadGuard rg(rw_lock_);
    if (segmented_store_) {
        segmented_store_->flush();
        return 0;
    }
    if (storage_) storage_->sync();
    if (persistence_manager) return persistence_manager->flush();
    return 0;
}

bool VectorDatabase::checkpoint() {
    if (segmented_store_) {
        segmented_store_->flush();
        return true;
    }
    if (storage_) {
        storage_->sync();
    }
    return true;
}

VectorDatabase::DatabaseStatistics VectorDatabase::getStatistics() const {
    RWLock::ReadGuard rg(rw_lock_);

    DatabaseStatistics stats{};
    stats.total_vectors = segmented_store_ ? segmented_store_->vectorCount() : key_to_slot_.size();
    stats.total_inserts = total_inserts.load(std::memory_order_relaxed);
    stats.total_searches = total_searches.load(std::memory_order_relaxed);
    stats.total_updates = total_updates.load(std::memory_order_relaxed);
    stats.total_deletes = total_deletes.load(std::memory_order_relaxed);
    stats.dimensions = dimensions;
    stats.search_mode = search_mode;
    stats.atomic_persistence_enabled = atomic_persistence_enabled;
    stats.batch_operations_enabled = batch_operations_enabled;
    stats.query_cache_enabled = query_cache_enabled;
    stats.storage_engine = storage_engine;

    if (persistence_manager) {
        stats.persistence_stats = persistence_manager->getStatistics();
    }
    if (batch_manager) {
        stats.batch_stats = batch_manager->getStatistics();
    }
    if (query_cache) {
        stats.cache_stats = query_cache->getStatistics();
    }
    if (segmented_store_) {
        stats.segmented_stats = segmented_store_->getStatistics();
    }

    return stats;
}

// -------------------- state helpers --------------------

bool VectorDatabase::isReady() const {
    if (!ready.load()) return false;
    if (persistence_manager) return !persistence_manager->isRecovering();
    return true;
}

bool VectorDatabase::isRecovering() const {
    if (persistence_manager) return persistence_manager->isRecovering();
    return false;
}

void VectorDatabase::updatePersistenceConfig(const PersistenceConfig& config) {
    RWLock::WriteGuard wg(rw_lock_);
    persistence_config = config;
    if (persistence_manager) {
        persistence_manager->updateConfig(config);
    }
}

std::unordered_map<std::string, Vector> VectorDatabase::getAllVectors() const {
    RWLock::ReadGuard rg(rw_lock_);
    if (segmented_store_) {
        return segmented_store_->getAllVectors();
    }
    std::unordered_map<std::string, Vector> result;
    result.reserve(key_to_slot_.size());
    for (const auto& [key, slot_id] : key_to_slot_) {
        const float* ptr = storage_->vector_ptr(slot_id);
        result.emplace(key, Vector(std::vector<float>(ptr, ptr + dimensions)));
    }
    return result;
}

const PersistenceConfig& VectorDatabase::getPersistenceConfig() const {
    return persistence_config;
}

void VectorDatabase::setReady(bool is_ready) {
    ready.store(is_ready);
}

void VectorDatabase::setRecovering(bool is_recovering) {
    recovering.store(is_recovering);
}

size_t VectorDatabase::vectorCount() const {
    RWLock::ReadGuard rg(rw_lock_);
    return segmented_store_ ? segmented_store_->vectorCount() : key_to_slot_.size();
}

void VectorDatabase::enableSIMD(bool enable) {
    Vector::enable_simd(enable);
}

bool VectorDatabase::isSIMDEnabled() const {
    return Vector::is_simd_enabled();
}

// -------------------- GPU acceleration --------------------

void VectorDatabase::enableGPU(bool enable) {
    RWLock::WriteGuard wg(rw_lock_);
    if (enable && !gpu_initialized) {
        if (gpu_ops::initialize()) {
            gpu_initialized = true;
            gpu_enabled = true;
        } else {
            gpu_enabled = false;
        }
    } else if (enable && gpu_initialized) {
        gpu_enabled = true;
    } else {
        gpu_enabled = false;
    }
}

bool VectorDatabase::isGPUEnabled() const {
    RWLock::ReadGuard rg(rw_lock_);
    return gpu_enabled;
}

bool VectorDatabase::isGPUAvailable() const {
    return gpu_ops::is_available();
}

void VectorDatabase::setGPUThreshold(size_t threshold) {
    RWLock::WriteGuard wg(rw_lock_);
    gpu_threshold = threshold;
}

size_t VectorDatabase::getGPUThreshold() const {
    RWLock::ReadGuard rg(rw_lock_);
    return gpu_threshold;
}

std::vector<std::pair<std::string, float>> VectorDatabase::gpuAcceleratedSearch(const Vector& query, size_t k) {
    std::lock_guard<std::mutex> gpu_lock(gpu_mutex_);

    if (gpu_buffer_dirty.load(std::memory_order_acquire)) {
        rebuildGPUBuffer();
    }

    std::vector<float> distances = gpu_ops::search_euclidean(query);

    if (distances.empty()) {
        return exactSearch(query, k);
    }

    std::vector<std::pair<size_t, float>> indexed;
    indexed.reserve(distances.size());
    for (size_t i = 0; i < distances.size(); ++i) {
        indexed.emplace_back(i, distances[i]);
    }

    size_t actual_k = std::min(k, indexed.size());
    std::ranges::partial_sort(indexed,
                              indexed.begin() + static_cast<std::ptrdiff_t>(actual_k),
                              {},
                              &std::pair<size_t, float>::second);

    std::vector<std::pair<std::string, float>> results;
    results.reserve(actual_k);
    for (size_t i = 0; i < actual_k; i++) {
        results.emplace_back(vector_keys[indexed[i].first], indexed[i].second);
    }

    return results;
}

void VectorDatabase::rebuildGPUBuffer() {
    flat_vectors.clear();
    vector_keys.clear();
    flat_vectors.reserve(key_to_slot_.size() * dimensions);
    vector_keys.reserve(key_to_slot_.size());

    for (const auto& [key, slot_id] : key_to_slot_) {
        vector_keys.push_back(key);
        const float* ptr = storage_->vector_ptr(slot_id);
        size_t offset = flat_vectors.size();
        flat_vectors.resize(offset + dimensions);
        std::memcpy(flat_vectors.data() + offset, ptr, dimensions * sizeof(float));
    }

    if (!flat_vectors.empty()) {
        gpu_ops::set_database_buffer(flat_vectors.data(), key_to_slot_.size(), dimensions);
    }

    gpu_buffer_dirty.store(false, std::memory_order_release);
}

void VectorDatabase::markGPUBufferDirty() {
    gpu_buffer_dirty.store(true, std::memory_order_release);
}
