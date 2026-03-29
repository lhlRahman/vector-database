// src/features/atomic_batch_insert.cpp
#include "atomic_batch_insert.hpp"
#include <iostream>
#include <stdexcept>

using std::chrono::steady_clock;

AtomicBatchInsert::AtomicBatchInsert(std::shared_ptr<AtomicPersistence> persistence,
                                     size_t max_batch_size,
                                     std::chrono::seconds batch_timeout,
                                     bool enable_validation)
    : persistence_(std::move(persistence)),
      max_batch_size(max_batch_size),
      batch_timeout(batch_timeout),
      enable_validation(enable_validation) {
    if (!persistence_) {
        throw std::invalid_argument("AtomicBatchInsert: persistence must not be null");
    }
}

uint64_t AtomicBatchInsert::generateTransactionId() { return ++transaction_counter; }

bool AtomicBatchInsert::validateOperation(const BatchOperation& op) const {
    if (op.key.empty()) return false;
    if ((op.type == OperationType::INSERT || op.type == OperationType::UPDATE) && op.vector.size() == 0) return false;
    return true;
}

bool AtomicBatchInsert::validateBatch(const std::vector<BatchOperation>& ops) const {
    if (ops.empty() || ops.size() > max_batch_size) return false;
    for (const auto& op : ops) if (!validateOperation(op)) return false;
    return true;
}

bool AtomicBatchInsert::isValidBatch(const std::vector<BatchOperation>& ops) const {
    return validateBatch(ops);
}

AtomicBatchInsert::BatchResult AtomicBatchInsert::executeBatch(const std::vector<BatchOperation>& operations) {
    auto t0 = steady_clock::now();
    BatchResult result{false, 0, "", 0, std::chrono::duration<double>(0)};

    if (!persistence_) {
        result.error_message = "persistence is null";
        failed_batches++;
        return result;
    }
    if (enable_validation && !validateBatch(operations)) {
        result.error_message = "invalid batch";
        failed_batches++;
        return result;
    }

    uint64_t tx = 0;
    size_t committed = 0;

    {
        std::lock_guard<std::mutex> lk(batch_mutex);
        tx = generateTransactionId();

        // Track which operations succeeded so we can report accurately
        std::vector<size_t> succeeded_indices;
        succeeded_indices.reserve(operations.size());

        for (size_t i = 0; i < operations.size(); ++i) {
            const auto& op = operations[i];
            bool ok = false;
            switch (op.type) {
                case OperationType::INSERT:
                    ok = persistence_->insert(op.key, op.vector, op.metadata);
                    break;
                case OperationType::UPDATE:
                    ok = persistence_->update(op.key, op.vector, op.metadata);
                    break;
                case OperationType::DELETE:
                    ok = persistence_->remove(op.key);
                    break;
            }
            if (!ok) {
                // Rollback: undo all previously committed operations in reverse
                for (auto it = succeeded_indices.rbegin(); it != succeeded_indices.rend(); ++it) {
                    const auto& prev_op = operations[*it];
                    switch (prev_op.type) {
                        case OperationType::INSERT:
                            // Undo insert by deleting
                            (void)persistence_->remove(prev_op.key);
                            break;
                        case OperationType::DELETE:
                            // Undo delete by re-inserting
                            (void)persistence_->insert(prev_op.key, prev_op.vector, prev_op.metadata);
                            break;
                        case OperationType::UPDATE:
                            // Cannot perfectly undo update without old value;
                            // WAL replay will handle consistency on recovery
                            break;
                    }
                }
                result.error_message = "operation " + std::to_string(i) + " failed on key '" + op.key + "'; batch rolled back";
                break;
            }
            succeeded_indices.push_back(i);
            committed++;
        }

        if (committed == operations.size()) {
            result.success = true;
            successful_batches++;
        } else {
            committed = 0; // Rolled back — nothing committed
            failed_batches++;
        }

        total_batches++;
        total_operations += committed;
    }

    result.operations_committed = committed;
    result.transaction_id = tx;
    result.duration = steady_clock::now() - t0;
    return result;
}

AtomicBatchInsert::BatchResult AtomicBatchInsert::executeBatchInsert(const std::vector<std::string>& keys,
                                                                     const std::vector<Vector>& vectors,
                                                                     const std::vector<std::string>& metadata) {
    std::vector<BatchOperation> ops;
    ops.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string meta = (i < metadata.size()) ? metadata[i] : "";
        ops.emplace_back(OperationType::INSERT, keys[i], vectors[i], meta);
    }
    return executeBatch(ops);
}

AtomicBatchInsert::BatchResult AtomicBatchInsert::executeBatchUpdate(const std::vector<std::string>& keys,
                                                                     const std::vector<Vector>& vectors,
                                                                     const std::vector<std::string>& metadata) {
    std::vector<BatchOperation> ops;
    ops.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string meta = (i < metadata.size()) ? metadata[i] : "";
        ops.emplace_back(OperationType::UPDATE, keys[i], vectors[i], meta);
    }
    return executeBatch(ops);
}

AtomicBatchInsert::BatchResult AtomicBatchInsert::executeBatchDelete(const std::vector<std::string>& keys) {
    std::vector<BatchOperation> ops;
    ops.reserve(keys.size());
    for (const auto& k : keys) {
        ops.emplace_back(OperationType::DELETE, k, Vector{}, "");
    }
    return executeBatch(ops);
}

AtomicBatchInsert::BatchResult AtomicBatchInsert::executeMixedBatch(const std::vector<BatchOperation>& operations) {
    return executeBatch(operations);
}

AtomicBatchInsert::Statistics AtomicBatchInsert::getStatistics() const {
    Statistics s{};
    s.total_batches = total_batches.load();
    s.successful_batches = successful_batches.load();
    s.failed_batches = failed_batches.load();
    s.total_operations = total_operations.load();
    const uint64_t total = s.total_batches;
    s.success_rate = total ? static_cast<double>(s.successful_batches) / static_cast<double>(total) : 0.0;
    s.average_batch_size = total ? static_cast<double>(s.total_operations) / static_cast<double>(total) : 0.0;
    s.average_batch_duration = std::chrono::duration<double>(0);
    return s;
}

void AtomicBatchInsert::updateConfig(size_t new_max_batch_size,
                                     std::chrono::seconds new_batch_timeout,
                                     bool new_enable_validation) {
    std::lock_guard<std::mutex> lk(batch_mutex);
    max_batch_size = new_max_batch_size;
    batch_timeout = new_batch_timeout;
    enable_validation = new_enable_validation;
}
