#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vector_database.hpp"
#include "../algorithms/flat_index.hpp"
#include "../optimizations/gpu_operations.hpp"
#include "../optimizations/simd_operations.hpp"

namespace {
constexpr size_t kMaxWalKeyBytes = 1u << 20;
constexpr size_t kMaxWalMetadataBytes = 16u << 20;

// A vector with any NaN component corrupts distance-based search (NaN poisons
// comparisons and, mixed with finite distances, can break the sort ordering).
// insert() has always rejected these; update()/batch paths must too.
bool containsNaN(const Vector& v) {
    return std::ranges::any_of(v, [](float f) { return std::isnan(f); });
}

size_t insertFrameBytes(const Vector& vector,
                        const std::string& key,
                        const std::string& metadata) {
    constexpr size_t header_bytes = 40;
    const size_t vector_bytes = vector.size() * sizeof(float);
    if (key.size() > std::numeric_limits<size_t>::max() - metadata.size() ||
        key.size() + metadata.size() >
            std::numeric_limits<size_t>::max() - vector_bytes - header_bytes) {
        return std::numeric_limits<size_t>::max();
    }
    return header_bytes + key.size() + metadata.size() + vector_bytes;
}
}  // namespace

struct VectorDatabase::RecallCommitterState {
    enum class RequestKind { Insert, Fence };

    struct Request {
        RequestKind kind{RequestKind::Insert};
        Vector vector;
        std::string key;
        std::string metadata;
        vdb::AckMode ack_mode{vdb::AckMode::Stable};
        std::promise<vdb::WriteReceipt> promise;
        uint64_t assigned_lsn{0};
        bool completed{false};
    };

    std::mutex mutex;
    std::condition_variable queue_cv;
    std::condition_variable durable_cv;
    std::deque<std::shared_ptr<Request>> queue;
    bool accepting{false};
    bool stop_requested{false};
    bool running{false};
    bool has_oldest_weak{false};
    std::chrono::steady_clock::time_point oldest_weak{};
    std::thread worker;

    vdb::RecallCommitConfig config;
    vdb::RecallCommitPolicyEvaluator policy;
    vdb::DurabilityFrontier frontier;
    vdb::CommitterHealth health{vdb::CommitterHealth::Healthy};
    RecallCommitterStatistics stats{};

    RecallCommitterState() : policy(config), frontier(0) {}
};

// -------------------- ctor / dtor --------------------

std::string VectorDatabase::make_temp_path() {
    // Both engines accept this path: MMap treats it as a file (writes
    // it directly), Segmented treats it as a directory (creates
    // segments/ + manifest.txt inside). The .vdb suffix is dropped
    // because for the segmented engine this is a directory name.
    auto tmp = std::filesystem::temp_directory_path();
    // The "vdb_auto_" prefix is the sentinel the destructor checks for
    // before doing best-effort cleanup. User-supplied paths (any other
    // prefix, or anything outside /tmp) are never auto-removed.
    auto name = "vdb_auto_" + std::to_string(getpid()) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
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
                               StorageEngine engine,
                               vdb::OpenMode open_mode)
    : storage_path_(storage_path.empty() ? make_temp_path() : storage_path),
      storage_engine(engine),
      open_mode_(open_mode),
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
    recall_committer_ = std::make_unique<RecallCommitterState>();
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
        config.hnsw_seed = hnsw_seed;
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
                distance_metric, vec_accessor_, hnsw_allocation_strategy,
                hnsw_arena_initial_size, hnsw_seed);
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
    // Best-effort cleanup of paths we ourselves auto-allocated under
    // /tmp/vdb_auto_<pid>_<ts>. Anything outside that exact temp-root +
    // basename pattern is treated as user-supplied and left alone.
    const auto storage_path = std::filesystem::path(storage_path_);
    const auto tmp_root = std::filesystem::temp_directory_path();
    const auto filename = storage_path.filename().string();
    if (storage_path.parent_path() == tmp_root &&
        filename.rfind("vdb_auto_", 0) == 0) {
        std::error_code ec;
        if (storage_engine == StorageEngine::Segmented) {
            std::filesystem::remove_all(storage_path_, ec);  // directory
        } else {
            std::filesystem::remove(storage_path_, ec);      // single file
        }
    }
}

// -------------------- lifecycle --------------------

void VectorDatabase::initialize() {
    bool start_committer = false;
    {
        RWLock::WriteGuard wg(rw_lock_);
        if (ready.load()) return;

        if (storage_engine == StorageEngine::Segmented) {
            segmented_store_->initialize(open_mode_ == vdb::OpenMode::ReadOnlyRecovery);
            recall_committer_->frontier.resetRecovered(segmented_store_->durableLsn());
            ready.store(true);
            start_committer = recall_committer_->config.enabled &&
                              open_mode_ == vdb::OpenMode::ReadWrite;
        } else {
            storage_->open();
            storage_->advise_sequential();
            key_to_slot_ = storage_->build_key_index();

            std::vector<std::pair<std::string, uint64_t>> ordered_slots(
                key_to_slot_.begin(), key_to_slot_.end());
            std::ranges::sort(ordered_slots, [](const auto& left, const auto& right) {
                if (left.second != right.second) return left.second < right.second;
                return left.first < right.first;
            });
            for (const auto& [key, slot_id] : ordered_slots) {
                if (hnsw_index) hnsw_index->insert(slot_id, key);
            }
            storage_->advise_random();

            if (atomic_persistence_enabled) initializeAtomicPersistence();
            quantizer_dirty_.store(true);
            if (!key_to_slot_.empty()) rebuildQuantizer();
            ready.store(true);
        }
    }
    if (start_committer) startRecallCommitter();
}

void VectorDatabase::shutdown() {
    if (!ready.load()) return;
    stopRecallCommitter(open_mode_ == vdb::OpenMode::ReadWrite);

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

void VectorDatabase::startRecallCommitter() {
    auto& committer = *recall_committer_;
    std::lock_guard lock(committer.mutex);
    if (committer.running) return;
    if (!committer.config.enabled || open_mode_ != vdb::OpenMode::ReadWrite) return;
    committer.accepting = true;
    committer.stop_requested = false;
    committer.health = vdb::CommitterHealth::Healthy;
    committer.running = true;
    committer.worker = std::thread([this] { recallCommitterLoop(); });
}

void VectorDatabase::stopRecallCommitter(bool fence) {
    auto& committer = *recall_committer_;
    std::future<vdb::WriteReceipt> fence_future;
    bool wait_for_fence = false;
    {
        std::lock_guard lock(committer.mutex);
        if (!committer.running) return;
        if (fence && committer.accepting &&
            committer.health == vdb::CommitterHealth::Healthy) {
            auto request = std::make_shared<RecallCommitterState::Request>();
            request->kind = RecallCommitterState::RequestKind::Fence;
            fence_future = request->promise.get_future();
            committer.queue.push_back(std::move(request));
            wait_for_fence = true;
            committer.queue_cv.notify_one();
        }
    }
    if (wait_for_fence) {
        try {
            (void)fence_future.get();
        } catch (...) {
            // The worker is still joined below; shutdown cannot report a false
            // durable frontier after a failed fence.
        }
    }

    {
        std::lock_guard lock(committer.mutex);
        committer.accepting = false;
        committer.stop_requested = true;
        if (committer.health == vdb::CommitterHealth::Healthy) {
            committer.health = vdb::CommitterHealth::ShuttingDown;
        }
        committer.queue_cv.notify_all();
    }
    if (committer.worker.joinable()) committer.worker.join();
    {
        std::lock_guard lock(committer.mutex);
        committer.running = false;
    }
}

void VectorDatabase::syncRecallFrontierFromStore() {
    if (!segmented_store_) return;
    const uint64_t visible = segmented_store_->visibleLsn();
    const uint64_t durable = segmented_store_->durableLsn();
    auto current = recall_committer_->frontier.snapshot();
    if (visible > current.visible_lsn) {
        recall_committer_->frontier.publishVisible(visible);
        current.visible_lsn = visible;
    }
    if (durable > current.durable_lsn) {
        recall_committer_->frontier.advanceDurable(durable);
        recall_committer_->durable_cv.notify_all();
    }
}

void VectorDatabase::requireWritable() const {
    if (open_mode_ == vdb::OpenMode::ReadOnlyRecovery) {
        throw std::logic_error("database is open in read-only recovery mode");
    }
}

void VectorDatabase::recallCommitterLoop() {
    auto& committer = *recall_committer_;
    using Request = RecallCommitterState::Request;

    for (;;) {
        std::vector<std::shared_ptr<Request>> requests;
        bool age_fence = false;
        {
            std::unique_lock lock(committer.mutex);
            while (committer.queue.empty() && !committer.stop_requested) {
                if (committer.has_oldest_weak &&
                    committer.config.max_tail_age.count() > 0) {
                    const auto deadline = committer.oldest_weak +
                                          committer.config.max_tail_age;
                    if (committer.queue_cv.wait_until(lock, deadline) ==
                        std::cv_status::timeout) {
                        age_fence = true;
                        break;
                    }
                } else {
                    committer.queue_cv.wait(lock);
                }
            }
            if (committer.has_oldest_weak &&
                committer.config.max_tail_age.count() > 0 &&
                std::chrono::steady_clock::now() >=
                    committer.oldest_weak + committer.config.max_tail_age) {
                age_fence = true;
            }
            if (committer.stop_requested && committer.queue.empty() && !age_fence) break;

            if (!age_fence && !committer.queue.empty() &&
                committer.config.group_delay.count() > 0 &&
                !committer.stop_requested) {
                committer.queue_cv.wait_for(lock, committer.config.group_delay, [&] {
                    return committer.stop_requested;
                });
            }
            while (!committer.queue.empty()) {
                requests.push_back(std::move(committer.queue.front()));
                committer.queue.pop_front();
            }
            if (requests.size() > 1) {
                committer.stats.follower_requests += requests.size() - 1;
            }
        }

        std::vector<std::pair<std::shared_ptr<Request>, vdb::AdmissionDecision>> stable;
        std::vector<std::shared_ptr<Request>> fence_waiters;
        try {
            RWLock::WriteGuard guard(rw_lock_);

            auto fence_now = [&](bool policy_fence, bool timed_fence) {
                const size_t weak_records = segmented_store_->volatileCount();
                if (weak_records == 0) return segmented_store_->durableLsn();
                const uint64_t target = segmented_store_->visibleLsn();
                {
                    std::lock_guard lock(committer.mutex);
                    ++committer.stats.sync_attempts;
                    if (policy_fence) ++committer.stats.policy_fences;
                    if (timed_fence) ++committer.stats.age_fences;
                }
                try {
                    // A durability fence must not change query representation.
                    // Seal/compaction are explicit maintenance operations while
                    // weak ACK mode is active.
                    segmented_store_->commitThrough(target, false);
                } catch (...) {
                    std::lock_guard lock(committer.mutex);
                    ++committer.stats.sync_failures;
                    throw;
                }
                committer.frontier.advanceDurable(target);
                {
                    std::lock_guard lock(committer.mutex);
                    ++committer.stats.sync_successes;
                    committer.stats.records_synced += weak_records;
                    committer.has_oldest_weak = false;
                }
                committer.durable_cv.notify_all();
                return target;
            };

            if (age_fence) fence_now(false, true);

            for (const auto& request : requests) {
                if (request->kind == RecallCommitterState::RequestKind::Fence) {
                    fence_waiters.push_back(request);
                    continue;
                }

                const size_t incoming_bytes =
                    insertFrameBytes(request->vector, request->key, request->metadata);
                auto admission_state = [&] {
                    const size_t weak = segmented_store_->volatileCount();
                    return vdb::AdmissionState{
                        segmented_store_->vectorCount() - weak,
                        weak,
                        segmented_store_->volatileBytes(),
                    };
                };
                vdb::AdmissionDecision decision = committer.policy.evaluateAdmission(
                    request->ack_mode, admission_state(), 1, incoming_bytes);
                if (decision.action == vdb::AdmissionAction::FenceThenRetry) {
                    fence_now(true, false);
                    decision = committer.policy.evaluateAdmission(
                        request->ack_mode, admission_state(), 1, incoming_bytes);
                }

                auto staged = segmented_store_->stageInsert(
                    request->vector, request->key, request->metadata);
                if (!staged.applied) {
                    vdb::WriteReceipt receipt;
                    receipt.requested_ack = request->ack_mode;
                    receipt.visible_lsn = segmented_store_->visibleLsn();
                    receipt.durable_lsn = segmented_store_->durableLsn();
                    receipt.durable_count = segmented_store_->vectorCount() -
                                            segmented_store_->volatileCount();
                    receipt.weak_count = segmented_store_->volatileCount();
                    request->promise.set_value(receipt);
                    request->completed = true;
                    continue;
                }

                committer.frontier.publishVisible(staged.lsn);
                request->assigned_lsn = staged.lsn;
                if (query_cache) query_cache->invalidate();
                total_inserts.fetch_add(1, std::memory_order_relaxed);

                if (decision.action == vdb::AdmissionAction::AdmitWeak) {
                    const size_t weak = segmented_store_->volatileCount();
                    vdb::WriteReceipt receipt{
                        true,
                        staged.lsn,
                        request->ack_mode,
                        vdb::AckLevel::Weak,
                        true,
                        segmented_store_->visibleLsn(),
                        segmented_store_->durableLsn(),
                        segmented_store_->vectorCount() - weak,
                        weak,
                        decision.policy_record_cap,
                        decision.estimated_recall_loss,
                        decision.correlation_alarm,
                    };
                    {
                        std::lock_guard lock(committer.mutex);
                        ++committer.stats.weak_acks;
                        committer.stats.max_weak_records =
                            std::max<uint64_t>(committer.stats.max_weak_records, weak);
                        if (!committer.has_oldest_weak) {
                            committer.oldest_weak = std::chrono::steady_clock::now();
                            committer.has_oldest_weak = true;
                        }
                    }
                    request->promise.set_value(receipt);
                    request->completed = true;
                } else {
                    stable.emplace_back(request, decision);
                    if (request->ack_mode == vdb::AckMode::Weak) {
                        std::lock_guard lock(committer.mutex);
                        ++committer.stats.auto_stable;
                    }
                }
            }

            if (!stable.empty() || !fence_waiters.empty()) {
                fence_now(false, false);
                std::lock_guard lock(committer.mutex);
                if (!fence_waiters.empty()) ++committer.stats.explicit_fences;
            }

            for (auto& [request, decision] : stable) {
                const size_t weak = segmented_store_->volatileCount();
                vdb::WriteReceipt receipt{
                    true,
                    request->assigned_lsn,
                    request->ack_mode,
                    vdb::AckLevel::Stable,
                    false,
                    segmented_store_->visibleLsn(),
                    segmented_store_->durableLsn(),
                    segmented_store_->vectorCount() - weak,
                    weak,
                    decision.policy_record_cap,
                    decision.estimated_recall_loss,
                    decision.correlation_alarm,
                };
                request->promise.set_value(receipt);
                request->completed = true;
                std::lock_guard lock(committer.mutex);
                ++committer.stats.stable_acks;
            }
            for (const auto& request : fence_waiters) {
                vdb::WriteReceipt receipt;
                receipt.applied = true;
                receipt.lsn = segmented_store_->durableLsn();
                receipt.requested_ack = vdb::AckMode::Stable;
                receipt.actual_ack = vdb::AckLevel::Stable;
                receipt.visible_lsn = segmented_store_->visibleLsn();
                receipt.durable_lsn = segmented_store_->durableLsn();
                receipt.durable_count = segmented_store_->vectorCount();
                request->promise.set_value(receipt);
                request->completed = true;
            }
        } catch (...) {
            const auto failure = std::current_exception();
            std::vector<std::shared_ptr<Request>> abandoned;
            {
                std::lock_guard lock(committer.mutex);
                committer.health = vdb::CommitterHealth::SyncFailed;
                committer.accepting = false;
                committer.stop_requested = true;
                while (!committer.queue.empty()) {
                    abandoned.push_back(std::move(committer.queue.front()));
                    committer.queue.pop_front();
                }
            }
            for (const auto& request : requests) {
                if (!request->completed) request->promise.set_exception(failure);
            }
            for (const auto& request : abandoned) {
                request->promise.set_exception(failure);
            }
            committer.durable_cv.notify_all();
            break;
        }
    }
}

void VectorDatabase::requestAsyncFence() {
    auto& committer = *recall_committer_;
    std::lock_guard lock(committer.mutex);
    if (!committer.running || !committer.accepting) return;
    const bool already_queued = std::ranges::any_of(committer.queue, [](const auto& request) {
        return request->kind == RecallCommitterState::RequestKind::Fence;
    });
    if (already_queued) return;
    auto request = std::make_shared<RecallCommitterState::Request>();
    request->kind = RecallCommitterState::RequestKind::Fence;
    committer.queue.push_back(std::move(request));
    committer.queue_cv.notify_one();
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
    if (ready.load()) requireWritable();
    if (storage_engine == StorageEngine::Segmented && ready.load() &&
        recall_committer_->config.enabled) {
        (void)durabilityFence();
    }
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

void VectorDatabase::configureHNSW(size_t M, size_t ef_construction, size_t ef_search,
                                   uint32_t seed) {
    RWLock::WriteGuard wg(rw_lock_);

    hnsw_M = M;
    hnsw_ef_construction = ef_construction;
    hnsw_ef_search = ef_search;
    hnsw_seed = seed;
    if (segmented_store_) {
        segmented_store_->configureHNSW(M, ef_construction, ef_search, seed);
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

void VectorDatabase::configureRecallCommit(const vdb::RecallCommitConfig& config) {
    vdb::RecallCommitPolicyEvaluator::validateConfig(config);
    RWLock::WriteGuard guard(rw_lock_);
    if (ready.load()) {
        throw std::logic_error("recall committer must be configured before initialize");
    }
    if (storage_engine != StorageEngine::Segmented && config.enabled) {
        throw std::invalid_argument("weak ACK requires the segmented storage engine");
    }
    {
        std::lock_guard lock(recall_committer_->mutex);
        recall_committer_->config = config;
        recall_committer_->policy.updateConfig(config);
    }
    hnsw_seed = config.hnsw_seed;
    if (segmented_store_) {
        segmented_store_->configureHNSW(
            hnsw_M, hnsw_ef_construction, hnsw_ef_search, config.hnsw_seed);
    }
}

vdb::WriteReceipt VectorDatabase::insertWithAck(
    const Vector& vector,
    const std::string& key,
    const std::string& metadata,
    vdb::AckMode ack_mode) {
    requireWritable();
    if (storage_engine != StorageEngine::Segmented) {
        throw std::invalid_argument("insertWithAck is supported only by segmented storage");
    }
    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");
    if (containsNaN(vector)) return {};
    if (key.size() > kMaxWalKeyBytes) {
        throw std::length_error("WAL key exceeds maximum encoded length");
    }
    if (metadata.size() > kMaxWalMetadataBytes) {
        throw std::length_error("WAL metadata exceeds maximum encoded length");
    }

    auto& committer = *recall_committer_;
    std::shared_ptr<RecallCommitterState::Request> request;
    std::future<vdb::WriteReceipt> future;
    {
        std::lock_guard lock(committer.mutex);
        if (committer.config.enabled) {
            if (!committer.running || !committer.accepting ||
                committer.health != vdb::CommitterHealth::Healthy) {
                throw std::runtime_error("recall committer is not accepting writes");
            }
            request = std::make_shared<RecallCommitterState::Request>();
            request->kind = RecallCommitterState::RequestKind::Insert;
            request->vector = vector;
            request->key = key;
            request->metadata = metadata;
            request->ack_mode = ack_mode;
            future = request->promise.get_future();
            committer.queue.push_back(request);
            committer.queue_cv.notify_one();
        }
    }
    if (request) return future.get();

    RWLock::WriteGuard guard(rw_lock_);
    const bool inserted = segmented_store_->insert(vector, key, metadata);
    syncRecallFrontierFromStore();
    if (inserted) {
        if (query_cache) query_cache->invalidate();
        total_inserts.fetch_add(1, std::memory_order_relaxed);
    }
    const size_t weak = segmented_store_->volatileCount();
    return vdb::WriteReceipt{
        inserted,
        inserted ? segmented_store_->durableLsn() : 0,
        ack_mode,
        inserted ? vdb::AckLevel::Stable : vdb::AckLevel::None,
        false,
        segmented_store_->visibleLsn(),
        segmented_store_->durableLsn(),
        segmented_store_->vectorCount() - weak,
        weak,
        0,
        0.0,
        false,
    };
}

uint64_t VectorDatabase::durabilityFence() {
    requireWritable();
    if (storage_engine != StorageEngine::Segmented) {
        RWLock::WriteGuard guard(rw_lock_);
        if (storage_) storage_->sync();
        return 0;
    }
    if (!ready.load()) throw std::runtime_error("Database not initialized");

    auto& committer = *recall_committer_;
    std::future<vdb::WriteReceipt> future;
    bool queued = false;
    {
        std::lock_guard lock(committer.mutex);
        if (committer.running) {
            if (!committer.accepting || committer.health != vdb::CommitterHealth::Healthy) {
                throw std::runtime_error("recall committer cannot fence");
            }
            auto request = std::make_shared<RecallCommitterState::Request>();
            request->kind = RecallCommitterState::RequestKind::Fence;
            future = request->promise.get_future();
            committer.queue.push_back(std::move(request));
            committer.queue_cv.notify_one();
            queued = true;
        }
    }
    if (queued) return future.get().durable_lsn;

    RWLock::WriteGuard guard(rw_lock_);
    const uint64_t visible = segmented_store_->visibleLsn();
    segmented_store_->commitThrough(
        visible, !recall_committer_->config.enabled);
    syncRecallFrontierFromStore();
    return segmented_store_->durableLsn();
}

bool VectorDatabase::waitUntilDurable(uint64_t lsn, std::chrono::milliseconds timeout) {
    if (lsn == 0) return false;
    auto& committer = *recall_committer_;
    std::unique_lock lock(committer.mutex);
    return committer.durable_cv.wait_for(lock, timeout, [&] {
        return committer.frontier.isDurable(lsn) ||
               committer.health == vdb::CommitterHealth::SyncFailed;
    }) && committer.frontier.isDurable(lsn);
}

vdb::DurabilityStatus VectorDatabase::durabilityStatus() const {
    RWLock::ReadGuard guard(rw_lock_);
    vdb::DurabilityStatus status;
    if (!segmented_store_) return status;

    const size_t weak = segmented_store_->volatileCount();
    const size_t visible = segmented_store_->vectorCount();
    const auto config = recall_committer_->policy.config();
    const auto correlation = recall_committer_->policy.correlationCounters();
    status.visible_lsn = segmented_store_->visibleLsn();
    status.appended_lsn = status.visible_lsn;
    status.durable_lsn = segmented_store_->durableLsn();
    status.visible_records = visible;
    status.durable_records = visible - weak;
    status.weak_records = weak;
    status.weak_bytes = segmented_store_->volatileBytes();
    status.policy_record_cap =
        vdb::RecallCommitPolicyEvaluator::policyRecordCap(config, status.durable_records);
    status.estimated_recall_loss =
        config.policy == vdb::RecallPolicy::Strict
            ? static_cast<double>(std::min(weak, config.k_min)) /
                  static_cast<double>(config.k_min)
            : vdb::RecallCommitPolicyEvaluator::expectedRecallLoss(
                  status.durable_records, weak);
    status.configured_policy = config.policy;
    status.effective_policy = correlation.alarmed ? vdb::RecallPolicy::Strict : config.policy;
    status.correlation_alarm = correlation.alarmed;
    status.manifest_generation = segmented_store_->manifestGeneration();
    {
        std::lock_guard lock(recall_committer_->mutex);
        status.health = recall_committer_->health;
    }
    return status;
}

VectorDatabase::RecallCommitterStatistics
VectorDatabase::recallCommitterStatistics() const {
    std::lock_guard lock(recall_committer_->mutex);
    return recall_committer_->stats;
}

vdb::RecallCommitPolicyCounters VectorDatabase::recallPolicyStatistics() const {
    return recall_committer_->policy.counters();
}

void VectorDatabase::sealMutableSegment() {
    requireWritable();
    if (segmented_store_ && ready.load() && recall_committer_->config.enabled) {
        (void)durabilityFence();
    }
    RWLock::WriteGuard wg(rw_lock_);
    if (segmented_store_) {
        segmented_store_->sealMutableSegment();
        if (query_cache) query_cache->clear();
    }
}

void VectorDatabase::compactSegments() {
    requireWritable();
    if (segmented_store_ && ready.load() && recall_committer_->config.enabled) {
        (void)durabilityFence();
    }
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
            distance_metric, vec_accessor_, hnsw_allocation_strategy,
            hnsw_arena_initial_size, hnsw_seed);
    }

    std::vector<std::pair<std::string, uint64_t>> ordered_slots(
        key_to_slot_.begin(), key_to_slot_.end());
    std::ranges::sort(ordered_slots, [](const auto& left, const auto& right) {
        if (left.second != right.second) return left.second < right.second;
        return left.first < right.first;
    });
    for (const auto& [key, slot_id] : ordered_slots) {
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
    if (storage_engine == StorageEngine::Segmented) {
        return insertWithAck(vector, key, metadata, vdb::AckMode::Stable).applied;
    }
    RWLock::WriteGuard wg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    if (std::ranges::any_of(vector, [](float f) { return std::isnan(f); })) {
        std::cerr << "Warning: Vector " << key << " contains NaN values. Skipping insertion.\n";
        return false;
    }

    if (key_to_slot_.count(key)) return false;

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
    requireWritable();
    if (storage_engine == StorageEngine::Segmented && recall_committer_->config.enabled) {
        (void)durabilityFence();
    }
    RWLock::WriteGuard wg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (vector.size() != dimensions) throw std::invalid_argument("Vector dimension mismatch");

    if (containsNaN(vector)) {
        std::cerr << "Warning: Vector " << key << " contains NaN values. Skipping update.\n";
        return false;
    }

    if (storage_engine == StorageEngine::Segmented) {
        bool updated = segmented_store_->update(vector, key, metadata);
        if (updated) {
            syncRecallFrontierFromStore();
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
    requireWritable();
    if (storage_engine == StorageEngine::Segmented && recall_committer_->config.enabled) {
        (void)durabilityFence();
    }
    RWLock::WriteGuard wg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");

    if (storage_engine == StorageEngine::Segmented) {
        bool removed = segmented_store_->remove(key);
        if (removed) {
            syncRecallFrontierFromStore();
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

std::optional<VectorDatabase::RecordSnapshot> VectorDatabase::inspectRecord(
    const std::string& key, vdb::ReadVisibility visibility) const {
    RWLock::ReadGuard guard(rw_lock_);
    if (storage_engine != StorageEngine::Segmented) {
        throw std::invalid_argument("record inspection requires segmented storage");
    }
    const auto record = segmented_store_->inspectRecord(
        key, visibility == vdb::ReadVisibility::Stable);
    if (!record) return std::nullopt;
    return RecordSnapshot{
        record->key, record->vector, record->metadata, record->lsn, record->provisional};
}

std::vector<VectorDatabase::RecordSnapshot> VectorDatabase::inspectRecords(
    vdb::ReadVisibility visibility) const {
    RWLock::ReadGuard guard(rw_lock_);
    if (storage_engine != StorageEngine::Segmented) {
        throw std::invalid_argument("record inspection requires segmented storage");
    }
    const auto records = segmented_store_->inspectRecords(
        visibility == vdb::ReadVisibility::Stable);
    std::vector<RecordSnapshot> snapshots;
    snapshots.reserve(records.size());
    for (const auto& record : records) {
        snapshots.push_back(RecordSnapshot{
            record.key, record.vector, record.metadata, record.lsn, record.provisional});
    }
    return snapshots;
}

VectorDatabase::SearchResponse VectorDatabase::similaritySearch(
    const Vector& query, size_t k, vdb::ReadVisibility visibility) {
    RWLock::ReadGuard guard(rw_lock_);
    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");
    if (storage_engine != StorageEngine::Segmented) {
        throw std::invalid_argument("visibility-aware search requires segmented storage");
    }

    SearchResponse response;
    size_t volatile_hits = 0;
    if (visibility == vdb::ReadVisibility::Stable) {
        for (const auto& [key, distance] : segmented_store_->searchStable(query, k)) {
            response.results.push_back(SearchResult{
                key, distance, segmented_store_->getMetadata(key)});
        }
        response.snapshot_lsn = segmented_store_->durableLsn();
    } else {
        for (const auto& result : segmented_store_->searchWithMetadata(query, k)) {
            response.results.push_back(SearchResult{
                result.key, result.distance, result.metadata});
            if (result.provisional) ++volatile_hits;
        }
        response.snapshot_lsn = segmented_store_->visibleLsn();
    }
    response.durable_lsn = segmented_store_->durableLsn();
    response.manifest_generation = segmented_store_->manifestGeneration();
    response.exact_tail_distance_evaluations = segmented_store_->volatileCount();
    total_searches.fetch_add(1, std::memory_order_relaxed);

    if (visibility == vdb::ReadVisibility::Latest && recall_committer_->config.enabled) {
        const auto before = recall_committer_->policy.correlationCounters().alarmed;
        recall_committer_->policy.observeQuery(
            segmented_store_->vectorCount(),
            segmented_store_->volatileCount(),
            response.results.size(),
            volatile_hits);
        const auto after = recall_committer_->policy.correlationCounters().alarmed;
        if (!before && after) requestAsyncFence();
    }
    return response;
}

std::vector<std::pair<std::string, float>> VectorDatabase::similaritySearch(const Vector& query, size_t k) {
    if (storage_engine == StorageEngine::Segmented) {
        auto response = similaritySearch(query, k, vdb::ReadVisibility::Latest);
        std::vector<std::pair<std::string, float>> results;
        results.reserve(response.results.size());
        for (const auto& result : response.results) {
            results.emplace_back(result.key, result.distance);
        }
        return results;
    }
    RWLock::ReadGuard rg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");

    if (key_to_slot_.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    std::vector<std::pair<std::string, float>> results;
    if (query_cache && query_cache->get(query, k, results)) {
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
        query_cache->put(query, k, results);
    }

    return results;
}

std::vector<VectorDatabase::SearchResult>
VectorDatabase::similaritySearchWithMetadata(const Vector& query, size_t k) {
    if (storage_engine == StorageEngine::Segmented) {
        return similaritySearch(query, k, vdb::ReadVisibility::Latest).results;
    }
    RWLock::ReadGuard rg(rw_lock_);

    if (!ready.load()) throw std::runtime_error("Database not initialized");
    if (query.size() != dimensions) throw std::invalid_argument("Query vector dimension mismatch");

    if (key_to_slot_.empty()) return {};

    total_searches.fetch_add(1, std::memory_order_relaxed);

    std::vector<std::pair<std::string, float>> rawResults;
    if (query_cache && query_cache->get(query, k, rawResults)) {
        // cache hit
    } else if (gpu_enabled && key_to_slot_.size() > gpu_threshold) {
        rawResults = gpuAcceleratedSearch(query, k);
    } else if (search_mode == SearchMode::HNSW && hnsw_index) {
        rawResults = hnsw_index->search(query, k);
    } else {
        rawResults = exactSearch(query, k);
    }
    if (query_cache) {
        query_cache->put(query, k, rawResults);
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
        if (query_cache && query_cache->get(query, k, single_result)) {
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
            query_cache->put(query, k, single_result);
        }

        results.push_back(std::move(single_result));
    }

    return results;
}

// -------------------- batch --------------------

AtomicBatchInsert::BatchResult VectorDatabase::batchInsert(const std::vector<std::string>& keys,
                                                           const std::vector<Vector>& vectors,
                                                           const std::vector<std::string>& metadata) {
    requireWritable();
    if (!batch_operations_enabled) {
        throw std::runtime_error("Batch operations not enabled");
    }

    if (keys.size() != vectors.size()) {
        return AtomicBatchInsert::BatchResult{false, 0, "Keys and vectors size mismatch", 0, std::chrono::duration<double>(0)};
    }
    if (storage_engine == StorageEngine::Segmented && recall_committer_->config.enabled) {
        (void)durabilityFence();
    }

    auto start_time = std::chrono::steady_clock::now();
    AtomicBatchInsert::BatchResult result;
    result.transaction_id = ++batch_transaction_counter;
    result.success = true;
    result.operations_committed = 0;

    {
        RWLock::WriteGuard wg(rw_lock_);

        if (storage_engine == StorageEngine::Segmented) {
            // Validate up front (dims/NaN — the single-insert path does this), then
            // group-commit the valid subset: one fsync for the whole batch instead
            // of one per row.
            std::vector<std::string> vkeys;
            std::vector<Vector> vvecs;
            std::vector<std::string> vmetas;
            vkeys.reserve(keys.size());
            vvecs.reserve(keys.size());
            vmetas.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) {
                if (vectors[i].size() != dimensions || containsNaN(vectors[i])) {
                    result.success = false;
                    result.error_message = "invalid vector (dims/NaN) for key: " + keys[i];
                    continue;
                }
                const std::string& meta = i < metadata.size() ? metadata[i] : "";
                if (keys[i].size() > kMaxWalKeyBytes ||
                    meta.size() > kMaxWalMetadataBytes) {
                    result.success = false;
                    result.error_message = "WAL payload exceeds encoded length limit";
                    continue;
                }
                vkeys.push_back(keys[i]);
                vvecs.push_back(vectors[i]);
                vmetas.push_back(meta);
            }
            result.operations_committed = segmented_store_->insertBatch(vkeys, vvecs, vmetas);
            syncRecallFrontierFromStore();
            if (query_cache) query_cache->invalidate();
            total_inserts.fetch_add(result.operations_committed, std::memory_order_relaxed);
            result.duration = std::chrono::steady_clock::now() - start_time;
            return result;
        }

        // Use sequential access for bulk insert
        storage_->advise_sequential();

        // Track committed inserts so we can roll the whole batch back on failure
        // (all-or-nothing), rather than leaving a partial, durable mutation.
        std::vector<std::pair<std::string, uint64_t>> committed;
        committed.reserve(keys.size());

        for (size_t i = 0; i < keys.size(); ++i) {
            const std::string& key = keys[i];
            const Vector& vector = vectors[i];
            const std::string& meta = (i < metadata.size()) ? metadata[i] : "";

            if (key_to_slot_.count(key)) continue;

            if (vector.size() != dimensions || containsNaN(vector)) {
                result.success = false;
                result.error_message = "invalid vector (dims/NaN) for key: " + key;
                break;
            }

            uint64_t slot_id = storage_->insert(key, vector.data_ptr(), meta);
            key_to_slot_[key] = slot_id;

            if (hnsw_index) hnsw_index->insert(slot_id, key);

            if (persistence_manager) {
                if (!persistence_manager->insert(key, vector, meta)) {
                    storage_->remove(slot_id);
                    key_to_slot_.erase(key);
                    if (hnsw_index) hnsw_index->remove(key);
                    result.success = false;
                    result.error_message = "Failed to persist key: " + key;
                    break;
                }
            }

            committed.emplace_back(key, slot_id);
            result.operations_committed++;
        }

        if (!result.success) {
            // Undo every committed insert in reverse so the batch is atomic.
            for (auto it = committed.rbegin(); it != committed.rend(); ++it) {
                storage_->remove(it->second);
                key_to_slot_.erase(it->first);
                if (hnsw_index) hnsw_index->remove(it->first);
                if (persistence_manager) (void)persistence_manager->remove(it->first);
            }
            result.operations_committed = 0;
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
    requireWritable();
    if (!batch_operations_enabled) {
        throw std::runtime_error("Batch operations not enabled");
    }

    if (keys.size() != vectors.size()) {
        return AtomicBatchInsert::BatchResult{false, 0, "Keys and vectors size mismatch", 0, std::chrono::duration<double>(0)};
    }
    if (storage_engine == StorageEngine::Segmented && recall_committer_->config.enabled) {
        (void)durabilityFence();
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
                if (vectors[i].size() != dimensions || containsNaN(vectors[i])) {
                    result.success = false;
                    result.error_message = "invalid vector (dims/NaN) for key: " + keys[i];
                    continue;
                }
                const std::string& meta = (i < metadata.size()) ? metadata[i] : "";
                if (segmented_store_->update(vectors[i], keys[i], meta)) {
                    result.operations_committed++;
                }
            }
            syncRecallFrontierFromStore();
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

            if (vector.size() != dimensions || containsNaN(vector)) {
                result.success = false;
                result.error_message = "invalid vector (dims/NaN) for key: " + key;
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
    requireWritable();
    if (!batch_operations_enabled) {
        throw std::runtime_error("Batch operations not enabled");
    }
    if (storage_engine == StorageEngine::Segmented && recall_committer_->config.enabled) {
        (void)durabilityFence();
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
            syncRecallFrontierFromStore();
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
    requireWritable();
    if (segmented_store_ && recall_committer_->config.enabled) {
        (void)durabilityFence();
    }
    RWLock::WriteGuard wg(rw_lock_);
    if (segmented_store_) {
        segmented_store_->flush();
        return 0;
    }
    if (storage_) storage_->sync();
    if (persistence_manager) return persistence_manager->flush();
    return 0;
}

bool VectorDatabase::checkpoint() {
    requireWritable();
    if (segmented_store_ && recall_committer_->config.enabled) {
        (void)durabilityFence();
    }
    // Flush/checkpoint mutate WAL and manifest state, so they serialize with
    // writers rather than sharing the read side of the database lock.
    RWLock::WriteGuard wg(rw_lock_);
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
        const size_t weak = stats.segmented_stats.volatile_records;
        stats.durability_status.visible_lsn = stats.segmented_stats.visible_lsn;
        stats.durability_status.appended_lsn = stats.segmented_stats.visible_lsn;
        stats.durability_status.durable_lsn = stats.segmented_stats.durable_lsn;
        stats.durability_status.visible_records = stats.segmented_stats.total_vectors;
        stats.durability_status.durable_records = stats.segmented_stats.total_vectors - weak;
        stats.durability_status.weak_records = weak;
        stats.durability_status.weak_bytes = segmented_store_->volatileBytes();
        stats.durability_status.manifest_generation =
            segmented_store_->manifestGeneration();
    }
    stats.committer_stats = recallCommitterStatistics();
    stats.policy_stats = recallPolicyStatistics();

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
    // NOTE: this toggles a PROCESS-GLOBAL flag (Vector::enable_simd), not per
    // instance — all VectorDatabase instances share it. It is currently inert
    // for search results: the distance metrics never consult it (its only
    // consumer, Vector::dot_product, has no production callers). Kept as a
    // global on purpose; documented so the per-instance-looking API isn't
    // mistaken for per-instance state.
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
    // The GPU kernel only implements Euclidean distance. For any other configured
    // metric, fall back to the metric-correct CPU path rather than silently
    // ranking by Euclidean.
    if (dynamic_cast<const EuclideanDistance*>(distance_metric.get()) == nullptr) {
        return exactSearch(query, k);
    }
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
