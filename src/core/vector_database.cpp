// Copyright [year] <Copyright Owner>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vector_database.hpp"

// -------------------- ctor / dtor --------------------

VectorDatabase::VectorDatabase(size_t dimensions,
                               const std::string& algorithm,
                               bool enable_atomic_persistence,
                               bool enable_batch_operations,
                               const PersistenceConfig& persistence_config)
    : dimensions(dimensions),
      approximate_algorithm(algorithm),
      atomic_persistence_enabled(enable_atomic_persistence),
      batch_operations_enabled(enable_batch_operations),
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
        lsh_index = std::make_unique<LSHIndex>(dimensions, 10, 8, distance_metric.get());
    } else if (algorithm == "hnsw") {
        hnsw_index = std::make_unique<HNSWIndex>(dimensions, 10, 8, 8, distance_metric.get());
    }
}

VectorDatabase::~VectorDatabase() {
    shutdown();
}

// -------------------- lifecycle --------------------

void VectorDatabase::initialize() {
    std::lock_guard<std::mutex> lock(db_mutex);

    if (ready.load()) return;

    std::cout << "Initializing Vector Database..." << std::endl;

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
              << vector_map.size() << " vectors." << std::endl;
}

void VectorDatabase::shutdown() {
    std::lock_guard<std::mutex> lock(db_mutex);

    if (!ready.load()) return;

    std::cout << "Shutting down Vector Database..." << std::endl;

    if (persistence_manager) {
        persistence_manager->shutdown();
    }

    ready.store(false);
    std::cout << "Vector Database shutdown completed" << std::endl;
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

    std::cout << "Loading existing data..." << std::endl;

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
                  << " vectors from persistent storage" << std::endl;
    }
}

// -------------------- configuration --------------------

void VectorDatabase::setDistanceMetric(std::shared_ptr<DistanceMetric> metric) {
    std::lock_guard<std::mutex> lock(db_mutex);

    distance_metric = std::move(metric);
    kd_tree = std::make_unique<KDTree>(dimensions, distance_metric);

    if (lsh_index) {
        lsh_index = std::make_unique<LSHIndex>(dimensions, 10, 8, distance_metric.get());
    }
    if (hnsw_index) {
        hnsw_index = std::make_unique<HNSWIndex>(dimensions, 10, 8, 8, distance_metric.get());
    }

    // Re-insert
    for (const auto& [key, vector] : vector_map) {
        kd_tree->insert(vector, key);
        if (lsh_index)  lsh_index->insert(vector, key);
        if (hnsw_index) hnsw_index->insert(vector, key);
    }
}

void VectorDatabase::setApproximateAlgorithm(const std::string& algorithm, size_t param1, size_t param2) {
    std::lock_guard<std::mutex> lock(db_mutex);

    approximate_algorithm = algorithm;

    lsh_index.reset();
    hnsw_index.reset();

    if (algorithm == "lsh") {
        lsh_index = std::make_unique<LSHIndex>(dimensions, param1, param2, distance_metric.get());
    } else if (algorithm == "hnsw") {
        hnsw_index = std::make_unique<HNSWIndex>(dimensions, param1, param2, param2, distance_metric.get());
    }

    for (const auto& [key, vector] : vector_map) {
        if (lsh_index)  lsh_index->insert(vector, key);
        if (hnsw_index) hnsw_index->insert(vector, key);
    }
}

// -------------------- mutations (with auto-checkpoint) --------------------

bool VectorDatabase::insert(const Vector& vector, const std::string& key, const std::string& metadata) {
    std::lock_guard<std::mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    for (size_t i = 0; i < vector.size(); ++i) {
        if (std::isnan(vector[i])) {
            std::cout << "Warning: Vector " << key << " contains NaN values. Skipping insertion." << std::endl;
            return false;
        }
    }

    // mutate in-memory first
    vector_map[key] = vector;
    if (!metadata.empty()) metadata_map[key] = metadata;

    kd_tree->insert(vector, key);
    if (lsh_index)  lsh_index->insert(vector, key);
    if (hnsw_index) hnsw_index->insert(vector, key);

    // durable WAL
    if (persistence_manager) {
        if (!persistence_manager->insert(key, vector, metadata)) {
            vector_map.erase(key);
            metadata_map.erase(key);
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

    total_inserts.fetch_add(1);
    return true;
}

bool VectorDatabase::insert(const Vector& vector, const std::string& key) {
    return insert(vector, key, "");
}

bool VectorDatabase::update(const Vector& vector, const std::string& key, const std::string& metadata) {
    std::lock_guard<std::mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector_map.find(key) == vector_map.end()) return false;
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    // mutate in-memory first
    vector_map[key] = vector;
    if (!metadata.empty()) metadata_map[key] = metadata;

    kd_tree->insert(vector, key);
    if (lsh_index)  lsh_index->insert(vector, key);
    if (hnsw_index) hnsw_index->insert(vector, key);

    // durable WAL
    if (persistence_manager) {
        if (!persistence_manager->update(key, vector, metadata)) {
            // NOTE: a real rollback would restore previous value
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

    total_updates.fetch_add(1);
    return true;
}

bool VectorDatabase::remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector_map.find(key) == vector_map.end()) return false;

    // mutate in-memory first
    vector_map.erase(key);
    metadata_map.erase(key);

    // durable WAL
    if (persistence_manager) {
        if (!persistence_manager->remove(key)) {
            // NOTE: a real rollback would restore previous value
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

    total_deletes.fetch_add(1);
    return true;
}

// -------------------- queries --------------------

const Vector* VectorDatabase::get(const std::string& key) const {
    std::lock_guard<std::mutex> lock(db_mutex);

    auto it = vector_map.find(key);
    if (it != vector_map.end()) return &it->second;

    if (persistence_manager) {
        auto [vector_ptr, metadata_ptr] = persistence_manager->get(key);
        if (vector_ptr) return vector_ptr;
    }

    return nullptr;
}

std::string VectorDatabase::getMetadata(const std::string& key) const {
    std::lock_guard<std::mutex> lock(db_mutex);

    auto it = metadata_map.find(key);
    if (it != metadata_map.end()) return it->second;

    return "";
}

std::vector<std::pair<std::string, float>> VectorDatabase::similaritySearch(const Vector& query, size_t k) {
    std::lock_guard<std::mutex> lock(db_mutex);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
    if (vector_map.empty()) return {};

    total_searches.fetch_add(1);

    if (approximate_algorithm == "lsh" && lsh_index) {
        return lsh_index->search(query, k);
    } else if (approximate_algorithm == "hnsw" && hnsw_index) {
        return hnsw_index->search(query, k);
    } else {
        return kd_tree->nearestNeighbors(query, k);
    }
}

std::vector<std::pair<std::string, float>>
VectorDatabase::similaritySearch(const Vector& query, size_t k) const {
    return const_cast<VectorDatabase*>(this)->similaritySearch(query, k);
}

std::vector<VectorDatabase::SearchResult>
VectorDatabase::similaritySearchWithMetadata(const Vector& query, size_t k) {
    auto rawResults = similaritySearch(query, k);
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
    std::vector<std::vector<std::pair<std::string, float>>> results;
    results.reserve(queries.size());

    for (const auto& query : queries) {
        results.push_back(similaritySearch(query, k));
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
    
    std::lock_guard<std::mutex> lock(db_mutex);
    
    auto start_time = std::chrono::steady_clock::now();
    AtomicBatchInsert::BatchResult result;
    result.transaction_id = ++batch_transaction_counter;  // You'll need to add this counter to the class
    result.success = true;
    result.operations_committed = 0;
    
    // Process each insert in the batch
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& key = keys[i];
        const Vector& vector = vectors[i];
        const std::string& meta = (i < metadata.size()) ? metadata[i] : "";
        
        // Check for duplicates
        if (vector_map.find(key) != vector_map.end()) {
            continue; // Skip duplicates, or you could update instead
        }
        
        // Validate vector
        if (vector.size() != dimensions) {
            result.success = false;
            result.error_message = "Vector dimension mismatch for key: " + key;
            break;
        }
        
        // Update in-memory structures
        vector_map[key] = vector;
        if (!meta.empty()) {
            metadata_map[key] = meta;
        }
        
        // Update indexes
        kd_tree->insert(vector, key);
        if (lsh_index) lsh_index->insert(vector, key);
        if (hnsw_index) hnsw_index->insert(vector, key);
        
        // Log to WAL for durability
        if (persistence_manager) {
            if (!persistence_manager->insert(key, vector, meta)) {
                // Rollback this operation
                vector_map.erase(key);
                metadata_map.erase(key);
                result.success = false;
                result.error_message = "Failed to persist key: " + key;
                break;
            }
        }
        
        result.operations_committed++;
    }
    
    // Update statistics
    if (result.success) {
        total_inserts.fetch_add(result.operations_committed);
    }
    
    // Check if we need to checkpoint after batch
    if (result.success && persistence_manager) {
        if (persistence_manager->shouldCheckpoint()) {
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
    
    std::lock_guard<std::mutex> lock(db_mutex);
    
    auto start_time = std::chrono::steady_clock::now();
    AtomicBatchInsert::BatchResult result;
    result.transaction_id = ++batch_transaction_counter;
    result.success = true;
    result.operations_committed = 0;
    
    // Process each update in the batch
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& key = keys[i];
        const Vector& vector = vectors[i];
        const std::string& meta = (i < metadata.size()) ? metadata[i] : "";
        
        // Check if exists
        if (vector_map.find(key) == vector_map.end()) {
            continue; // Skip non-existent keys for update
        }
        
        // Validate vector
        if (vector.size() != dimensions) {
            result.success = false;
            result.error_message = "Vector dimension mismatch for key: " + key;
            break;
        }
        
        // Update in-memory structures
        vector_map[key] = vector;
        if (!meta.empty()) {
            metadata_map[key] = meta;
        }
        
        // Update indexes
        kd_tree->insert(vector, key);
        if (lsh_index) lsh_index->insert(vector, key);
        if (hnsw_index) hnsw_index->insert(vector, key);
        
        // Log to WAL for durability
        if (persistence_manager) {
            if (!persistence_manager->update(key, vector, meta)) {
                result.success = false;
                result.error_message = "Failed to persist update for key: " + key;
                break;
            }
        }
        
        result.operations_committed++;
    }
    
    // Update statistics
    if (result.success) {
        total_updates.fetch_add(result.operations_committed);
    }
    
    // Check if we need to checkpoint after batch
    if (result.success && persistence_manager) {
        if (persistence_manager->shouldCheckpoint()) {
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
    
    std::lock_guard<std::mutex> lock(db_mutex);
    
    auto start_time = std::chrono::steady_clock::now();
    AtomicBatchInsert::BatchResult result;
    result.transaction_id = ++batch_transaction_counter;
    result.success = true;
    result.operations_committed = 0;
    
    // Process each delete in the batch
    for (const std::string& key : keys) {
        // Check if exists
        if (vector_map.find(key) == vector_map.end()) {
            continue; // Skip non-existent keys
        }
        
        // Remove from in-memory structures
        vector_map.erase(key);
        metadata_map.erase(key);
        
        // Log to WAL for durability
        if (persistence_manager) {
            if (!persistence_manager->remove(key)) {
                result.success = false;
                result.error_message = "Failed to persist deletion for key: " + key;
                break;
            }
        }
        
        result.operations_committed++;
    }
    
    // Update statistics
    if (result.success) {
        total_deletes.fetch_add(result.operations_committed);
    }
    
    // Check if we need to checkpoint after batch
    if (result.success && persistence_manager) {
        if (persistence_manager->shouldCheckpoint()) {
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
    if (persistence_manager) return persistence_manager->flush();
    return 0;
}

bool VectorDatabase::checkpoint() {
    if (persistence_manager) {
        bool success = persistence_manager->saveDatabase(vector_map, metadata_map);
        if (success) {
            persistence_manager->onCheckpointCompleted();
        }
        return success;
    }
    return true;
}

VectorDatabase::DatabaseStatistics VectorDatabase::getStatistics() const {
    std::lock_guard<std::mutex> lock(db_mutex);

    DatabaseStatistics stats;
    stats.total_vectors = vector_map.size();
    stats.total_inserts = total_inserts.load();
    stats.total_searches = total_searches.load();
    stats.total_updates = total_updates.load();
    stats.total_deletes = total_deletes.load();
    stats.dimensions = dimensions;
    stats.algorithm = approximate_algorithm;
    stats.atomic_persistence_enabled = atomic_persistence_enabled;
    stats.batch_operations_enabled = batch_operations_enabled;

    if (persistence_manager) {
        stats.persistence_stats = persistence_manager->getStatistics();
    }
    if (batch_manager) {
        stats.batch_stats = batch_manager->getStatistics();
    }

    return stats;
}

// -------------------- state helpers --------------------

RecoveryStateMachine::RecoveryInfo VectorDatabase::getRecoveryInfo() const {
    if (persistence_manager) return persistence_manager->getRecoveryInfo();
    return RecoveryStateMachine::RecoveryInfo();
}

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
    std::lock_guard<std::mutex> lock(db_mutex);

    persistence_config = config;

    if (persistence_manager) {
        persistence_manager->updateConfig(config);
    }
}

const std::unordered_map<std::string, Vector>& VectorDatabase::getAllVectors() const {
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

std::unordered_map<std::string, Vector> VectorDatabase::getAllVectorsCopy() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(db_mutex));
    return vector_map;
}