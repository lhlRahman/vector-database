#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vector_database.hpp"
#include "../optimizations/gpu_operations.hpp"

// -------------------- ctor / dtor --------------------

VectorDatabase::VectorDatabase(size_t dimensions,
                               const std::string& algorithm,
                               bool enable_atomic_persistence,
                               bool enable_batch_operations,
                               const PersistenceConfig& persistence_config,
                               bool enable_query_cache,
                               size_t cache_capacity)
    : dimensions(dimensions),
      approximate_algorithm(algorithm),
      atomic_persistence_enabled(enable_atomic_persistence),
      batch_operations_enabled(enable_batch_operations),
      query_cache_enabled(enable_query_cache),
      persistence_config(persistence_config),
      total_inserts(0),
      total_searches(0),
      total_updates(0),
      total_deletes(0) {
    // Distance metric
    distance_metric = std::make_shared<EuclideanDistance>();
    // KD-tree
    kd_tree = std::make_unique<KDTree>(dimensions, distance_metric);

    // Approximate indexes
    if (algorithm == "lsh") {
        lsh_num_tables = kDefaultLSHTables;
        lsh_num_hash_functions = kDefaultLSHHashFunctions;
        lsh_index = std::make_unique<LSHIndex>(dimensions, lsh_num_tables, lsh_num_hash_functions, distance_metric);
    } else if (algorithm == "hnsw") {
        hnsw_M = kDefaultHNSW_M;
        hnsw_ef_construction = kDefaultHNSW_EfConstruction;
        hnsw_ef_search = kDefaultHNSW_EfSearch;
        hnsw_index = std::make_unique<HNSWIndex>(dimensions, hnsw_M, hnsw_ef_construction, hnsw_ef_search, distance_metric);
    }
    
    // Query cache
    if (enable_query_cache) {
        query_cache = std::make_unique<QueryCache>(cache_capacity);
    }
}

VectorDatabase::~VectorDatabase() noexcept {
    try {
        shutdown();
    } catch (...) {
        // Destructors must not throw
    }
}

// -------------------- lifecycle --------------------

void VectorDatabase::initialize() {
    std::unique_lock<std::shared_mutex> lock(db_mutex);

    if (ready.load()) return;

    std::cout << "Initializing Vector Database..." << '\n';

    if (atomic_persistence_enabled) {
        initializeAtomicPersistence();

        if (!persistence_manager) {
            throw std::runtime_error("persistence_manager is null after initialization");
        }

        // Recovery
        setRecovering(true);
        if (!persistence_manager->loadDatabase(vector_map, metadata_map)) {
            setRecovering(false);
            throw std::runtime_error("Failed to recover database from persistent storage.");
        }
        setRecovering(false);

        // Rebuild in-memory indexes
        for (const auto& [key, vector] : vector_map) {
            kd_tree->insert(vector, key);
            if (lsh_index)  lsh_index->insert(vector, key);
            if (hnsw_index) hnsw_index->insert(vector, key);
        }
    }

    ready.store(true);
    std::cout << "Vector Database initialized successfully with "
              << vector_map.size() << " vectors." << '\n';
}

void VectorDatabase::shutdown() {
    std::unique_lock<std::shared_mutex> lock(db_mutex);

    if (!ready.load()) return;

    std::cout << "Shutting down Vector Database..." << '\n';

    if (persistence_manager) {
        persistence_manager->shutdown();
    }

    ready.store(false);
    std::cout << "Vector Database shutdown completed" << '\n';
}

void VectorDatabase::initializeAtomicPersistence() {
    // Shared ownership
    persistence_manager = std::make_shared<AtomicPersistence>(persistence_config);
    persistence_manager->initialize();

    if (batch_operations_enabled) {
        // IMPORTANT: share, do NOT move (keeps persistence_manager alive)
        batch_manager = std::make_unique<AtomicBatchInsert>(persistence_manager);
    }
}

void VectorDatabase::loadExistingData() {
    if (!persistence_manager) return;

    std::cout << "Loading existing data..." << '\n';

    std::unordered_map<std::string, Vector> loaded_vectors;
    std::unordered_map<std::string, std::string> loaded_metadata;

    if (persistence_manager->loadDatabase(loaded_vectors, loaded_metadata)) {
        for (const auto& [key, vector] : loaded_vectors) {
            vector_map[key] = vector;

            kd_tree->insert(vector, key);

            if (lsh_index)  lsh_index->insert(vector, key);
            if (hnsw_index) hnsw_index->insert(vector, key);
        }

        metadata_map = std::move(loaded_metadata);

        std::cout << "Loaded " << loaded_vectors.size()
                  << " vectors from persistent storage" << '\n';
    }
}

// -------------------- configuration --------------------

void VectorDatabase::setDistanceMetric(std::shared_ptr<DistanceMetric> metric) {
    std::unique_lock<std::shared_mutex> lock(db_mutex);

    distance_metric = std::move(metric);
    rebuildIndexes();

    if (query_cache) {
        query_cache->clear();
    }
}

void VectorDatabase::setApproximateAlgorithm(const std::string& algorithm, size_t param1, size_t param2) {
    std::unique_lock<std::shared_mutex> lock(db_mutex);

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
    kd_tree = std::make_unique<KDTree>(dimensions, distance_metric);
    lsh_index.reset();
    hnsw_index.reset();

    if (approximate_algorithm == "lsh") {
        lsh_index = std::make_unique<LSHIndex>(dimensions, lsh_num_tables, lsh_num_hash_functions, distance_metric);
    } else if (approximate_algorithm == "hnsw") {
        hnsw_index = std::make_unique<HNSWIndex>(dimensions, hnsw_M, hnsw_ef_construction, hnsw_ef_search, distance_metric);
    }

    for (const auto& [key, vector] : vector_map) {
        kd_tree->insert(vector, key);
        if (lsh_index)  lsh_index->insert(vector, key);
        if (hnsw_index) hnsw_index->insert(vector, key);
    }
}

// -------------------- mutations (with auto-checkpoint) --------------------

bool VectorDatabase::insert(const Vector& vector, const std::string& key, const std::string& metadata) {
    std::unique_lock<std::shared_mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    for (size_t i = 0; i < vector.size(); ++i) {
        if (std::isnan(vector[i])) {
            std::cerr << "Warning: Vector " << key << " contains NaN values. Skipping insertion.\n";
            return false;
        }
    }

    // Mutate in-memory first — single lookup via try_emplace
    auto [it, inserted] = vector_map.try_emplace(key, vector);
    if (!inserted) return false;
    if (!metadata.empty()) metadata_map[key] = metadata;

    try {
        kd_tree->insert(vector, key);
        if (lsh_index)  lsh_index->insert(vector, key);
        if (hnsw_index) hnsw_index->insert(vector, key);
    } catch (...) {
        // Rollback on index failure
        vector_map.erase(key);
        metadata_map.erase(key);
        throw;
    }

    // Invalidate query cache
    if (query_cache) {
        query_cache->invalidate();
    }

    markGPUBufferDirty();

    // Durable WAL
    if (persistence_manager) {
        if (!persistence_manager->insert(key, vector, metadata)) {
            vector_map.erase(key);
            metadata_map.erase(key);
            kd_tree->remove(key);
            if (lsh_index) lsh_index->remove(key);
            if (hnsw_index) hnsw_index->remove(key);
            return false;
        }

        // AUTO-CHECKPOINT: Check if we should checkpoint
        if (persistence_manager->shouldCheckpoint()) {
            bool checkpoint_success = persistence_manager->saveDatabase(vector_map, metadata_map);
            if (checkpoint_success) {
                // CRITICAL: Reset the operations counter after successful checkpoint
                persistence_manager->onCheckpointCompleted();
            }
        }
    }

    total_inserts.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool VectorDatabase::update(const Vector& vector, const std::string& key, const std::string& metadata) {
    std::unique_lock<std::shared_mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    auto it = vector_map.find(key);
    if (it == vector_map.end()) return false;

    // Save old state for rollback
    Vector old_vector = it->second;
    auto meta_it = metadata_map.find(key);
    std::string old_metadata = (meta_it != metadata_map.end()) ? meta_it->second : "";

    // Mutate in-memory
    it->second = vector;
    if (!metadata.empty()) metadata_map[key] = metadata;

    // Incremental index update: remove old, insert new
    try {
        kd_tree->remove(key);
        kd_tree->insert(vector, key);
        if (lsh_index) { lsh_index->remove(key); lsh_index->insert(vector, key); }
        if (hnsw_index) { hnsw_index->remove(key); hnsw_index->insert(vector, key); }
    } catch (...) {
        // Rollback on index failure
        it->second = old_vector;
        if (old_metadata.empty()) metadata_map.erase(key);
        else metadata_map[key] = old_metadata;
        throw;
    }

    // Invalidate query cache
    if (query_cache) {
        query_cache->invalidate();
    }

    markGPUBufferDirty();

    // Durable WAL
    if (persistence_manager) {
        if (!persistence_manager->update(key, vector, metadata)) {
            // Rollback in-memory state
            it->second = old_vector;
            if (old_metadata.empty()) metadata_map.erase(key);
            else metadata_map[key] = old_metadata;
            kd_tree->remove(key);
            kd_tree->insert(old_vector, key);
            if (lsh_index) { lsh_index->remove(key); lsh_index->insert(old_vector, key); }
            if (hnsw_index) { hnsw_index->remove(key); hnsw_index->insert(old_vector, key); }
            return false;
        }

        if (persistence_manager->shouldCheckpoint()) {
            bool checkpoint_success = persistence_manager->saveDatabase(vector_map, metadata_map);
            if (checkpoint_success) {
                persistence_manager->onCheckpointCompleted();
            }
        }
    }

    total_updates.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool VectorDatabase::remove(const std::string& key) {
    std::unique_lock<std::shared_mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");

    auto it = vector_map.find(key);
    if (it == vector_map.end()) return false;

    // Save old state for rollback
    Vector old_vector = it->second;
    auto meta_it = metadata_map.find(key);
    std::string old_metadata = (meta_it != metadata_map.end()) ? meta_it->second : "";

    // Mutate in-memory
    vector_map.erase(it);
    metadata_map.erase(key);

    // Lazy deletion from indexes
    kd_tree->remove(key);
    if (lsh_index) lsh_index->remove(key);
    if (hnsw_index) hnsw_index->remove(key);

    // Invalidate query cache
    if (query_cache) {
        query_cache->invalidate();
    }

    markGPUBufferDirty();

    // Durable WAL
    if (persistence_manager) {
        if (!persistence_manager->remove(key)) {
            // Rollback: re-insert into all structures
            vector_map[key] = old_vector;
            if (!old_metadata.empty()) metadata_map[key] = old_metadata;
            kd_tree->insert(old_vector, key);
            if (lsh_index) lsh_index->insert(old_vector, key);
            if (hnsw_index) hnsw_index->insert(old_vector, key);
            return false;
        }

        if (persistence_manager->shouldCheckpoint()) {
            bool checkpoint_success = persistence_manager->saveDatabase(vector_map, metadata_map);
            if (checkpoint_success) {
                persistence_manager->onCheckpointCompleted();
            }
        }
    }

    total_deletes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// -------------------- queries --------------------

std::optional<Vector> VectorDatabase::get(const std::string& key) const {
    std::shared_lock<std::shared_mutex> lock(db_mutex);

    auto it = vector_map.find(key);
    if (it != vector_map.end()) return it->second;

    return std::nullopt;
}

std::string VectorDatabase::getMetadata(const std::string& key) const {
    std::shared_lock<std::shared_mutex> lock(db_mutex);

    auto it = metadata_map.find(key);
    if (it != metadata_map.end()) return it->second;

    return "";
}

std::vector<std::pair<std::string, float>> VectorDatabase::similaritySearch(const Vector& query, size_t k) {
    std::shared_lock<std::shared_mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
    if (vector_map.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    // Try cache first
    std::vector<std::pair<std::string, float>> results;
    if (query_cache && query_cache->get(query, results)) {
        // Cache hit - return cached results
        return results;
    }

    // Use GPU for large datasets (brute-force on GPU is faster than CPU indexes)
    if (gpu_enabled && vector_map.size() > gpu_threshold) {
        results = gpuAcceleratedSearch(query, k);
    }
    // Cache miss - perform actual search using CPU indexes
    else if (approximate_algorithm == "lsh" && lsh_index) {
        results = lsh_index->search(query, k);
    } else if (approximate_algorithm == "hnsw" && hnsw_index) {
        results = hnsw_index->search(query, k);
    } else {
        results = kd_tree->nearestNeighbors(query, k);
    }
    
    // Store in cache for future queries
    if (query_cache) {
        query_cache->put(query, results);
    }
    
    return results;
}

std::vector<VectorDatabase::SearchResult>
VectorDatabase::similaritySearchWithMetadata(const Vector& query, size_t k) {
    std::shared_lock<std::shared_mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
    if (vector_map.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    // Perform search (same logic as similaritySearch but under one lock)
    std::vector<std::pair<std::string, float>> rawResults;
    if (query_cache && query_cache->get(query, rawResults)) {
        // cache hit
    } else if (gpu_enabled && vector_map.size() > gpu_threshold) {
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

    // Build results with metadata under the same lock
    std::vector<SearchResult> results;
    results.reserve(rawResults.size());
    for (const auto& [key, distance] : rawResults) {
        auto metaIt = metadata_map.find(key);
        results.emplace_back(SearchResult{
            key, distance, (metaIt != metadata_map.end() ? metaIt->second : "")
        });
    }

    return results;
}

std::vector<std::vector<std::pair<std::string, float>>>
VectorDatabase::batchSimilaritySearch(const std::vector<Vector>& queries, size_t k) {
    std::shared_lock<std::shared_mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");

    std::vector<std::vector<std::pair<std::string, float>>> results;
    results.reserve(queries.size());

    for (const auto& query : queries) {
        if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
        if (vector_map.empty()) { results.emplace_back(); continue; }

        total_searches.fetch_add(1, std::memory_order_relaxed);

        std::vector<std::pair<std::string, float>> single_result;
        if (query_cache && query_cache->get(query, single_result)) {
            results.push_back(std::move(single_result));
            continue;
        }

        if (gpu_enabled && vector_map.size() > gpu_threshold) {
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
        std::unique_lock<std::shared_mutex> lock(db_mutex);

        // Pre-reserve to avoid rehashes during batch
        vector_map.reserve(vector_map.size() + keys.size());
        metadata_map.reserve(metadata_map.size() + keys.size());

        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string& key = keys[i];
            const Vector& vector = vectors[i];
            const std::string& meta = (i < metadata.size()) ? metadata[i] : "";

            if (vector_map.find(key) != vector_map.end()) {
                continue;
            }

            if (vector.size() != dimensions) {
                result.success = false;
                result.error_message = "Vector dimension mismatch for key: " + key;
                break;
            }

            vector_map[key] = vector;
            if (!meta.empty()) {
                metadata_map[key] = meta;
            }

            kd_tree->insert(vector, key);
            if (lsh_index) lsh_index->insert(vector, key);
            if (hnsw_index) hnsw_index->insert(vector, key);

            if (persistence_manager) {
                if (!persistence_manager->insert(key, vector, meta)) {
                    vector_map.erase(key);
                    metadata_map.erase(key);
                    result.success = false;
                    result.error_message = "Failed to persist key: " + key;
                    break;
                }
            }

            result.operations_committed++;
        }

        if (query_cache) {
            query_cache->invalidate();
        }

        markGPUBufferDirty();
    } // lock released before checkpoint I/O

    // Stats outside lock
    if (result.success) {
        total_inserts.fetch_add(result.operations_committed, std::memory_order_relaxed);
    }

    // Checkpoint outside exclusive lock (saveDatabase acquires its own lock)
    if (result.success && persistence_manager) {
        if (persistence_manager->shouldCheckpoint()) {
            std::shared_lock<std::shared_mutex> read_lock(db_mutex);
            bool checkpoint_success = persistence_manager->saveDatabase(vector_map, metadata_map);
            if (checkpoint_success) {
                persistence_manager->onCheckpointCompleted();
            }
        }
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
        std::unique_lock<std::shared_mutex> lock(db_mutex);

        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string& key = keys[i];
            const Vector& vector = vectors[i];
            const std::string& meta = (i < metadata.size()) ? metadata[i] : "";

            if (vector_map.find(key) == vector_map.end()) {
                continue;
            }

            if (vector.size() != dimensions) {
                result.success = false;
                result.error_message = "Vector dimension mismatch for key: " + key;
                break;
            }

            vector_map[key] = vector;
            if (!meta.empty()) {
                metadata_map[key] = meta;
            }

            kd_tree->remove(key);
            kd_tree->insert(vector, key);
            if (lsh_index) { lsh_index->remove(key); lsh_index->insert(vector, key); }
            if (hnsw_index) { hnsw_index->remove(key); hnsw_index->insert(vector, key); }

            if (persistence_manager) {
                if (!persistence_manager->update(key, vector, meta)) {
                    result.success = false;
                    result.error_message = "Failed to persist update for key: " + key;
                    break;
                }
            }

            result.operations_committed++;
        }

        if (query_cache) {
            query_cache->invalidate();
        }
        markGPUBufferDirty();
    }

    if (result.success) {
        total_updates.fetch_add(result.operations_committed, std::memory_order_relaxed);
    }

    if (result.success && persistence_manager) {
        if (persistence_manager->shouldCheckpoint()) {
            std::shared_lock<std::shared_mutex> read_lock(db_mutex);
            bool checkpoint_success = persistence_manager->saveDatabase(vector_map, metadata_map);
            if (checkpoint_success) {
                persistence_manager->onCheckpointCompleted();
            }
        }
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
        std::unique_lock<std::shared_mutex> lock(db_mutex);

        for (const std::string& key : keys) {
            if (vector_map.find(key) == vector_map.end()) {
                continue;
            }

            vector_map.erase(key);
            metadata_map.erase(key);

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

        if (query_cache) {
            query_cache->invalidate();
        }
        markGPUBufferDirty();
    }

    if (result.success) {
        total_deletes.fetch_add(result.operations_committed, std::memory_order_relaxed);
    }

    if (result.success && persistence_manager) {
        if (persistence_manager->shouldCheckpoint()) {
            std::shared_lock<std::shared_mutex> read_lock(db_mutex);
            bool checkpoint_success = persistence_manager->saveDatabase(vector_map, metadata_map);
            if (checkpoint_success) {
                persistence_manager->onCheckpointCompleted();
            }
        }
    }

    result.duration = std::chrono::steady_clock::now() - start_time;
    return result;
}

// -------------------- maintenance / stats --------------------

size_t VectorDatabase::flush() {
    std::shared_lock<std::shared_mutex> lock(db_mutex);
    if (persistence_manager) return persistence_manager->flush();
    return 0;
}

bool VectorDatabase::checkpoint() {
    // Snapshot data under read lock, then write outside lock
    std::unordered_map<std::string, Vector> snapshot_vectors;
    std::unordered_map<std::string, std::string> snapshot_metadata;

    {
        std::shared_lock<std::shared_mutex> lock(db_mutex);
        if (!persistence_manager) return true;
        snapshot_vectors = vector_map;
        snapshot_metadata = metadata_map;
    }

    // I/O outside lock — doesn't block inserts/updates
    bool success = persistence_manager->saveDatabase(snapshot_vectors, snapshot_metadata);
    if (success) {
        persistence_manager->onCheckpointCompleted();
    }
    return success;
}

VectorDatabase::DatabaseStatistics VectorDatabase::getStatistics() const {
    std::shared_lock<std::shared_mutex> lock(db_mutex);

    DatabaseStatistics stats;
    stats.total_vectors = vector_map.size();
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
    std::unique_lock<std::shared_mutex> lock(db_mutex);

    persistence_config = config;

    if (persistence_manager) {
        persistence_manager->updateConfig(config);
    }
}

std::unordered_map<std::string, Vector> VectorDatabase::getAllVectors() const {
    std::shared_lock<std::shared_mutex> lock(db_mutex);
    return vector_map;
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
    std::shared_lock<std::shared_mutex> lock(db_mutex);
    return vector_map.size();
}

void VectorDatabase::enableSIMD(bool enable) {
    Vector::enable_simd(enable);
}

bool VectorDatabase::isSIMDEnabled() const {
    return Vector::is_simd_enabled();
}

// -------------------- GPU acceleration --------------------

void VectorDatabase::enableGPU(bool enable) {
    std::unique_lock<std::shared_mutex> lock(db_mutex);
    if (enable && !gpu_initialized) {
        if (gpu_ops::initialize()) {
            gpu_initialized = true;
            gpu_enabled = true;
            std::cout << "GPU acceleration enabled\n";
        } else {
            std::cerr << "Failed to initialize GPU, falling back to CPU\n";
            gpu_enabled = false;
        }
    } else if (enable && gpu_initialized) {
        gpu_enabled = true;
        std::cout << "GPU acceleration enabled\n";
    } else {
        gpu_enabled = false;
        std::cout << "GPU acceleration disabled\n";
    }
}

bool VectorDatabase::isGPUEnabled() const {
    std::shared_lock<std::shared_mutex> lock(db_mutex);
    return gpu_enabled;
}

bool VectorDatabase::isGPUAvailable() const {
    return gpu_ops::is_available();
}

void VectorDatabase::setGPUThreshold(size_t threshold) {
    std::unique_lock<std::shared_mutex> lock(db_mutex);
    gpu_threshold = threshold;
}

size_t VectorDatabase::getGPUThreshold() const {
    std::shared_lock<std::shared_mutex> lock(db_mutex);
    return gpu_threshold;
}

std::vector<std::pair<std::string, float>> VectorDatabase::gpuAcceleratedSearch(const Vector& query, size_t k) {
    std::lock_guard<std::mutex> gpu_lock(gpu_mutex_);

    // Rebuild GPU buffer if dirty (only happens after insert/update/delete)
    if (gpu_buffer_dirty.load(std::memory_order_acquire)) {
        rebuildGPUBuffer();
    }

    // Use the zero-copy GPU search
    std::vector<float> distances = gpu_ops::search_euclidean(query);

    // If GPU failed, fall back to CPU
    if (distances.empty()) {
        return kd_tree->nearestNeighbors(query, k);
    }

    // Find top-k using partial sort
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

    // Convert to key-distance pairs using our cached keys
    std::vector<std::pair<std::string, float>> results;
    results.reserve(actual_k);
    for (size_t i = 0; i < actual_k; i++) {
        results.emplace_back(vector_keys[indexed[i].first], indexed[i].second);
    }

    return results;
}

void VectorDatabase::rebuildGPUBuffer() {
    // Clear and rebuild contiguous storage
    flat_vectors.clear();
    vector_keys.clear();
    flat_vectors.reserve(vector_map.size() * dimensions);
    vector_keys.reserve(vector_map.size());

    for (const auto& [key, vec] : vector_map) {
        vector_keys.push_back(key);
        // Bulk copy via memcpy instead of float-by-float push_back
        size_t offset = flat_vectors.size();
        flat_vectors.resize(offset + dimensions);
        std::memcpy(flat_vectors.data() + offset, vec.data_ptr(), dimensions * sizeof(float));
    }

    // Update GPU buffer (zero-copy on Apple Silicon!)
    if (!flat_vectors.empty()) {
        gpu_ops::set_database_buffer(flat_vectors.data(), vector_map.size(), dimensions);
    }

    gpu_buffer_dirty.store(false, std::memory_order_release);
}

void VectorDatabase::markGPUBufferDirty() {
    gpu_buffer_dirty.store(true, std::memory_order_release);
}