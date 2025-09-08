// src/features/atomic_batch_insert.cpp
#include "atomic_batch_insert.hpp"
#include <cassert>
#include <iostream>

using std::chrono::steady_clock;

AtomicBatchInsert::AtomicBatchInsert(std::shared_ptr<AtomicPersistence> persistence,
                                     size_t max_batch_size,
                                     std::chrono::seconds batch_timeout,
                                     bool enable_validation)
    : persistence_(std::move(persistence)),
      max_batch_size(max_batch_size),
      batch_timeout(batch_timeout),
      enable_validation(enable_validation) {
    assert(persistence_ && "AtomicBatchInsert: persistence must not be null");
}

uint64_t AtomicBatchInsert::generateTransactionId() { return ++transaction_counter; }

void AtomicBatchInsert::logTransactionStart(uint64_t tx, size_t count) {
    (void)tx; (void)count;
}

void AtomicBatchInsert::logTransactionEnd(uint64_t tx, bool ok, const std::string& err) {
    (void)tx; (void)ok; (void)err;
}

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

    std::unique_lock<std::mutex> lk(batch_mutex);
    const uint64_t tx = generateTransactionId();
    logTransactionStart(tx, operations.size());

    size_t committed = 0;
    for (const auto& op : operations) {
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
            result.error_message = "operation failed";
            break;
        }
        committed++;
    }

    if (committed == operations.size()) {
        result.success = true;
        successful_batches++;
    } else {
        failed_batches++;
    }

    total_batches++;
    total_operations += committed;

    lk.unlock();

    result.operations_committed = committed;
    result.transaction_id = tx;
    result.duration = steady_clock::now() - t0;
    logTransactionEnd(tx, result.success, result.error_message);
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
    s.success_rate = total ? (double)s.successful_batches / (double)total : 0.0;
    s.average_batch_size = total ? (double)s.total_operations / (double)total : 0.0;
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
