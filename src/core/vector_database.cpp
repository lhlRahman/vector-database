#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vector_database.hpp"
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
                               const std::string& algorithm,
                               bool enable_atomic_persistence,
                               bool enable_batch_operations,
                               const PersistenceConfig& persistence_config,
                               bool enable_query_cache,
                               size_t cache_capacity,
                               const std::string& storage_path)
    : storage_path_(storage_path.empty() ? make_temp_path() : storage_path),
      dimensions(dimensions),
      approximate_algorithm(algorithm),
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

    // Vector accessor — all indexes use this to read from mmap
    vec_accessor_ = [this](uint64_t slot_id) -> const float* {
        return storage_->vector_ptr(slot_id);
    };

    // KD-tree
    kd_tree = std::make_unique<KDTree>(dimensions, distance_metric, vec_accessor_);

    // Approximate indexes
    if (algorithm == "lsh") {
        lsh_num_tables = kDefaultLSHTables;
        lsh_num_hash_functions = kDefaultLSHHashFunctions;
        lsh_index = std::make_unique<LSHIndex>(dimensions, lsh_num_tables, lsh_num_hash_functions, distance_metric, vec_accessor_);
    } else if (algorithm == "hnsw") {
        hnsw_M = kDefaultHNSW_M;
        hnsw_ef_construction = kDefaultHNSW_EfConstruction;
        hnsw_ef_search = kDefaultHNSW_EfSearch;
        hnsw_index = std::make_unique<HNSWIndex>(dimensions, hnsw_M, hnsw_ef_construction, hnsw_ef_search, distance_metric, vec_accessor_);
    }

    // Query cache
    if (enable_query_cache) {
        query_cache = std::make_unique<QueryCache>(cache_capacity);
    }

    // mmap storage
    storage_ = std::make_unique<MMapStorage>(storage_path_, dimensions);
}

VectorDatabase::~VectorDatabase() noexcept {
    try {
        shutdown();
    } catch (...) {}
    if (storage_path_.find("vdb_") != std::string::npos &&
        storage_path_.find(std::filesystem::temp_directory_path().string()) != std::string::npos) {
        std::filesystem::remove(storage_path_);
    }
}

// -------------------- lifecycle --------------------

void VectorDatabase::initialize() {
    EpochRCU::WriteGuard wg(rcu_);

    if (ready.load()) return;

    std::cout << "Initializing Vector Database..." << '\n';

    // Open mmap storage
    storage_->open();

    // Use sequential access pattern for bulk loading
    storage_->advise_sequential();

    // Rebuild key→slot index from mmap'd data
    key_to_slot_ = storage_->build_key_index();

    // Rebuild in-memory indexes from mmap'd vectors (using slot IDs, zero-copy)
    for (const auto& [key, slot_id] : key_to_slot_) {
        kd_tree->insert(slot_id, key);
        if (lsh_index)  lsh_index->insert(slot_id, key);
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
    std::cout << "Vector Database initialized successfully with "
              << key_to_slot_.size() << " vectors." << '\n';
}

void VectorDatabase::shutdown() {
    EpochRCU::WriteGuard wg(rcu_);

    if (!ready.load()) return;

    std::cout << "Shutting down Vector Database..." << '\n';

    if (persistence_manager) {
        persistence_manager->shutdown();
    }

    if (storage_) {
        storage_->sync();
        storage_->close();
    }

    ready.store(false);
    std::cout << "Vector Database shutdown completed" << '\n';
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

            kd_tree->insert(slot, key);
            if (lsh_index)  lsh_index->insert(slot, key);
            if (hnsw_index) hnsw_index->insert(slot, key);
        }

        std::cout << "Loaded " << loaded_vectors.size()
                  << " vectors from persistent storage" << '\n';
    }
}

// -------------------- configuration --------------------

void VectorDatabase::setDistanceMetric(std::shared_ptr<DistanceMetric> metric) {
    EpochRCU::WriteGuard wg(rcu_);

    distance_metric = std::move(metric);
    rebuildIndexes();

    if (query_cache) {
        query_cache->clear();
    }
}

void VectorDatabase::setApproximateAlgorithm(const std::string& algorithm, size_t param1, size_t param2) {
    EpochRCU::WriteGuard wg(rcu_);

    approximate_algorithm = algorithm;

    if (algorithm == "lsh") {
        lsh_num_tables = param1;
        lsh_num_hash_functions = param2;
    } else if (algorithm == "hnsw") {
        hnsw_M = param1;
        hnsw_ef_construction = param2;
        hnsw_ef_search = param2;
    }

    rebuildIndexes();

    if (query_cache) {
        query_cache->clear();
    }
}

void VectorDatabase::rebuildIndexes() {
    // Use sequential access for bulk rebuild
    storage_->advise_sequential();

    kd_tree = std::make_unique<KDTree>(dimensions, distance_metric, vec_accessor_);
    lsh_index.reset();
    hnsw_index.reset();

    if (approximate_algorithm == "lsh") {
        lsh_index = std::make_unique<LSHIndex>(dimensions, lsh_num_tables, lsh_num_hash_functions, distance_metric, vec_accessor_);
    } else if (approximate_algorithm == "hnsw") {
        hnsw_index = std::make_unique<HNSWIndex>(dimensions, hnsw_M, hnsw_ef_construction, hnsw_ef_search, distance_metric, vec_accessor_);
    }

    for (const auto& [key, slot_id] : key_to_slot_) {
        kd_tree->insert(slot_id, key);
        if (lsh_index)  lsh_index->insert(slot_id, key);
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
    EpochRCU::WriteGuard wg(rcu_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    for (size_t i = 0; i < vector.size(); ++i) {
        if (std::isnan(vector[i])) {
            std::cerr << "Warning: Vector " << key << " contains NaN values. Skipping insertion.\n";
            return false;
        }
    }

    if (key_to_slot_.count(key)) return false;

    uint64_t slot_id = storage_->insert(key, vector.data_ptr(), metadata);
    key_to_slot_[key] = slot_id;

    try {
        kd_tree->insert(slot_id, key);
        if (lsh_index)  lsh_index->insert(slot_id, key);
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
            kd_tree->remove(key);
            if (lsh_index) lsh_index->remove(key);
            if (hnsw_index) hnsw_index->remove(key);
            return false;
        }
    }

    total_inserts.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool VectorDatabase::update(const Vector& vector, const std::string& key, const std::string& metadata) {
    EpochRCU::WriteGuard wg(rcu_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return false;

    uint64_t slot_id = it->second;

    const float* old_ptr = storage_->vector_ptr(slot_id);
    std::vector<float> old_data(old_ptr, old_ptr + dimensions);
    std::string old_metadata = storage_->get_metadata(slot_id);

    storage_->update(slot_id, vector.data_ptr(), metadata);

    try {
        kd_tree->remove(key);
        kd_tree->insert(slot_id, key);
        if (lsh_index) { lsh_index->remove(key); lsh_index->insert(slot_id, key); }
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
            kd_tree->remove(key); kd_tree->insert(slot_id, key);
            if (lsh_index) { lsh_index->remove(key); lsh_index->insert(slot_id, key); }
            if (hnsw_index) { hnsw_index->remove(key); hnsw_index->insert(slot_id, key); }
            return false;
        }
    }

    total_updates.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool VectorDatabase::remove(const std::string& key) {
    EpochRCU::WriteGuard wg(rcu_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return false;

    uint64_t slot_id = it->second;

    const float* old_ptr = storage_->vector_ptr(slot_id);
    std::vector<float> old_data(old_ptr, old_ptr + dimensions);
    std::string old_metadata = storage_->get_metadata(slot_id);

    storage_->remove(slot_id);
    key_to_slot_.erase(it);

    kd_tree->remove(key);
    if (lsh_index) lsh_index->remove(key);
    if (hnsw_index) hnsw_index->remove(key);

    if (query_cache) query_cache->invalidate();
    markGPUBufferDirty();
    quantizer_dirty_.store(true);

    if (persistence_manager) {
        if (!persistence_manager->remove(key)) {
            uint64_t new_slot = storage_->insert(key, old_data.data(), old_metadata);
            key_to_slot_[key] = new_slot;
            kd_tree->insert(new_slot, key);
            if (lsh_index) lsh_index->insert(new_slot, key);
            if (hnsw_index) hnsw_index->insert(new_slot, key);
            return false;
        }
    }

    total_deletes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// -------------------- queries --------------------

std::optional<Vector> VectorDatabase::get(const std::string& key) const {
    EpochRCU::ReadGuard rg(rcu_);

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return std::nullopt;

    const float* ptr = storage_->vector_ptr(it->second);
    return Vector(std::vector<float>(ptr, ptr + dimensions));
}

std::string VectorDatabase::getMetadata(const std::string& key) const {
    EpochRCU::ReadGuard rg(rcu_);

    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return "";

    return storage_->get_metadata(it->second);
}

std::vector<std::pair<std::string, float>> VectorDatabase::similaritySearch(const Vector& query, size_t k) {
    EpochRCU::ReadGuard rg(rcu_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
    if (key_to_slot_.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    std::vector<std::pair<std::string, float>> results;
    if (query_cache && query_cache->get(query, results)) {
        return results;
    }

    if (gpu_enabled && key_to_slot_.size() > gpu_threshold) {
        results = gpuAcceleratedSearch(query, k);
    }
    else if (approximate_algorithm == "lsh" && lsh_index) {
        results = lsh_index->search(query, k);
    } else if (approximate_algorithm == "hnsw" && hnsw_index) {
        results = hnsw_index->search(query, k);
    } else {
        results = kd_tree->nearestNeighbors(query, k);
    }

    if (query_cache) {
        query_cache->put(query, results);
    }

    return results;
}

std::vector<VectorDatabase::SearchResult>
VectorDatabase::similaritySearchWithMetadata(const Vector& query, size_t k) {
    EpochRCU::ReadGuard rg(rcu_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
    if (key_to_slot_.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    std::vector<std::pair<std::string, float>> rawResults;
    if (query_cache && query_cache->get(query, rawResults)) {
        // cache hit
    } else if (gpu_enabled && key_to_slot_.size() > gpu_threshold) {
        rawResults = gpuAcceleratedSearch(query, k);
    } else if (approximate_algorithm == "lsh" && lsh_index) {
        rawResults = lsh_index->search(query, k);
    } else if (approximate_algorithm == "hnsw" && hnsw_index) {
        rawResults = hnsw_index->search(query, k);
    } else {
        rawResults = kd_tree->nearestNeighbors(query, k);
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
    EpochRCU::ReadGuard rg(rcu_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");

    std::vector<std::vector<std::pair<std::string, float>>> results;
    results.reserve(queries.size());

    for (const auto& query : queries) {
        if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
        if (key_to_slot_.empty()) { results.emplace_back(); continue; }

        total_searches.fetch_add(1, std::memory_order_relaxed);

        std::vector<std::pair<std::string, float>> single_result;
        if (query_cache && query_cache->get(query, single_result)) {
            results.push_back(std::move(single_result));
            continue;
        }

        if (gpu_enabled && key_to_slot_.size() > gpu_threshold) {
            single_result = gpuAcceleratedSearch(query, k);
        } else if (approximate_algorithm == "lsh" && lsh_index) {
            single_result = lsh_index->search(query, k);
        } else if (approximate_algorithm == "hnsw" && hnsw_index) {
            single_result = hnsw_index->search(query, k);
        } else {
            single_result = kd_tree->nearestNeighbors(query, k);
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
        EpochRCU::WriteGuard wg(rcu_);

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

            kd_tree->insert(slot_id, key);
            if (lsh_index) lsh_index->insert(slot_id, key);
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
        EpochRCU::WriteGuard wg(rcu_);

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

            kd_tree->remove(key);
            kd_tree->insert(it->second, key);
            if (lsh_index) { lsh_index->remove(key); lsh_index->insert(it->second, key); }
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
        EpochRCU::WriteGuard wg(rcu_);

        for (const std::string& key : keys) {
            auto it = key_to_slot_.find(key);
            if (it == key_to_slot_.end()) continue;

            storage_->remove(it->second);
            key_to_slot_.erase(it);

            kd_tree->remove(key);
            if (lsh_index) lsh_index->remove(key);
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
    EpochRCU::ReadGuard rg(rcu_);
    if (storage_) storage_->sync();
    if (persistence_manager) return persistence_manager->flush();
    return 0;
}

bool VectorDatabase::checkpoint() {
    if (storage_) {
        storage_->sync();
    }
    return true;
}

VectorDatabase::DatabaseStatistics VectorDatabase::getStatistics() const {
    EpochRCU::ReadGuard rg(rcu_);

    DatabaseStatistics stats;
    stats.total_vectors = key_to_slot_.size();
    stats.total_inserts = total_inserts.load(std::memory_order_relaxed);
    stats.total_searches = total_searches.load(std::memory_order_relaxed);
    stats.total_updates = total_updates.load(std::memory_order_relaxed);
    stats.total_deletes = total_deletes.load(std::memory_order_relaxed);
    stats.dimensions = dimensions;
    stats.algorithm = approximate_algorithm;
    stats.atomic_persistence_enabled = atomic_persistence_enabled;
    stats.batch_operations_enabled = batch_operations_enabled;
    stats.query_cache_enabled = query_cache_enabled;

    if (persistence_manager) {
        stats.persistence_stats = persistence_manager->getStatistics();
    }
    if (batch_manager) {
        stats.batch_stats = batch_manager->getStatistics();
    }
    if (query_cache) {
        stats.cache_stats = query_cache->getStatistics();
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
    EpochRCU::WriteGuard wg(rcu_);
    persistence_config = config;
    if (persistence_manager) {
        persistence_manager->updateConfig(config);
    }
}

std::unordered_map<std::string, Vector> VectorDatabase::getAllVectors() const {
    EpochRCU::ReadGuard rg(rcu_);
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
    EpochRCU::ReadGuard rg(rcu_);
    return key_to_slot_.size();
}

void VectorDatabase::enableSIMD(bool enable) {
    Vector::enable_simd(enable);
}

bool VectorDatabase::isSIMDEnabled() const {
    return Vector::is_simd_enabled();
}

// -------------------- GPU acceleration --------------------

void VectorDatabase::enableGPU(bool enable) {
    EpochRCU::WriteGuard wg(rcu_);
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
    EpochRCU::ReadGuard rg(rcu_);
    return gpu_enabled;
}

bool VectorDatabase::isGPUAvailable() const {
    return gpu_ops::is_available();
}

void VectorDatabase::setGPUThreshold(size_t threshold) {
    EpochRCU::WriteGuard wg(rcu_);
    gpu_threshold = threshold;
}

size_t VectorDatabase::getGPUThreshold() const {
    EpochRCU::ReadGuard rg(rcu_);
    return gpu_threshold;
}

std::vector<std::pair<std::string, float>> VectorDatabase::gpuAcceleratedSearch(const Vector& query, size_t k) {
    std::lock_guard<std::mutex> gpu_lock(gpu_mutex_);

    if (gpu_buffer_dirty.load(std::memory_order_acquire)) {
        rebuildGPUBuffer();
    }

    std::vector<float> distances = gpu_ops::search_euclidean(query);

    if (distances.empty()) {
        return kd_tree->nearestNeighbors(query, k);
    }

    std::vector<std::pair<size_t, float>> indexed;
    indexed.reserve(distances.size());
    for (size_t i = 0; i < distances.size(); i++) {
        indexed.emplace_back(i, distances[i]);
    }

    size_t actual_k = std::min(k, indexed.size());
    std::partial_sort(indexed.begin(),
                      indexed.begin() + static_cast<std::ptrdiff_t>(actual_k),
                      indexed.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });

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
