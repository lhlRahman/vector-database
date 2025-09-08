// src/features/atomic_batch_insert.hpp
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/vector.hpp"
#include "atomic_persistence.hpp"
#include "json.hpp"

using json = nlohmann::json;

class AtomicBatchInsert {
public:
    enum class OperationType { INSERT, UPDATE, DELETE };

    struct BatchOperation {
        OperationType type;
        std::string key;
        Vector vector;
        std::string metadata;
        std::chrono::steady_clock::time_point timestamp;

        BatchOperation(OperationType t, const std::string& k, const Vector& v, const std::string& m = "")
            : type(t), key(k), vector(v), metadata(m), timestamp(std::chrono::steady_clock::now()) {}
    };

    struct BatchResult {
        bool success;
        size_t operations_committed;
        std::string error_message;
        uint64_t transaction_id;
        std::chrono::duration<double> duration;
    };

private:
    std::shared_ptr<AtomicPersistence> persistence_;
    std::mutex batch_mutex;
    std::atomic<uint64_t> transaction_counter{0};

    // Batch configuration
    size_t max_batch_size;
    std::chrono::seconds batch_timeout;
    bool enable_validation;

    // Statistics
    std::atomic<uint64_t> total_batches{0};
    std::atomic<uint64_t> successful_batches{0};
    std::atomic<uint64_t> failed_batches{0};
    std::atomic<uint64_t> total_operations{0};

    // Validation
    bool validateBatch(const std::vector<BatchOperation>& operations) const;
    bool validateOperation(const BatchOperation& operation) const;

    // Tx mgmt
    uint64_t generateTransactionId();
    void logTransactionStart(uint64_t transaction_id, size_t operation_count);
    void logTransactionEnd(uint64_t transaction_id, bool success, const std::string& error = "");

public:
    // Declaration only (definition in .cpp)
    AtomicBatchInsert(std::shared_ptr<AtomicPersistence> persistence,
                      size_t max_batch_size = 10000,
                      std::chrono::seconds batch_timeout = std::chrono::seconds(30),
                      bool enable_validation = true);
    ~AtomicBatchInsert() = default;

    AtomicBatchInsert(const AtomicBatchInsert&) = delete;
    AtomicBatchInsert& operator=(const AtomicBatchInsert&) = delete;

    BatchResult executeBatch(const std::vector<BatchOperation>& operations);
    BatchResult executeBatchInsert(const std::vector<std::string>& keys,
                                   const std::vector<Vector>& vectors,
                                   const std::vector<std::string>& metadata = {});
    BatchResult executeBatchUpdate(const std::vector<std::string>& keys,
                                   const std::vector<Vector>& vectors,
                                   const std::vector<std::string>& metadata = {});
    BatchResult executeBatchDelete(const std::vector<std::string>& keys);
    BatchResult executeMixedBatch(const std::vector<BatchOperation>& operations);

    struct Statistics {
        uint64_t total_batches;
        uint64_t successful_batches;
        uint64_t failed_batches;
        uint64_t total_operations;
        double success_rate;
        double average_batch_size;
        std::chrono::duration<double> average_batch_duration;
    };

    Statistics getStatistics() const;

    void updateConfig(size_t max_batch_size,
                      std::chrono::seconds batch_timeout,
                      bool enable_validation);

    bool isValidBatch(const std::vector<BatchOperation>& operations) const;

    size_t getMaxBatchSize() const { return max_batch_size; }
    std::chrono::seconds getBatchTimeout() const { return batch_timeout; }
    bool isValidationEnabled() const { return enable_validation; }

    std::shared_ptr<AtomicPersistence> persistenceHandle() const { return persistence_; }
};

inline void to_json(json& j, const AtomicBatchInsert::Statistics& stats) {
    j = json{
        {"total_batches", stats.total_batches},
        {"successful_batches", stats.successful_batches},
        {"failed_batches", stats.failed_batches},
        {"total_operations", stats.total_operations},
        {"success_rate", stats.success_rate},
        {"average_batch_size", stats.average_batch_size},
        {"average_batch_duration", stats.average_batch_duration.count()}
    };
}
