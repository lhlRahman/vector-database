#include <atomic>
#include <cmath>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "../src/features/recall_commit_policy.hpp"
#include "../src/core/vector_database.hpp"

namespace {

int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        throw std::runtime_error(std::string("check failed: ") + #expr); \
    } \
} while (false)

#define CHECK_NEAR(a, b, tolerance) do { \
    const double lhs = static_cast<double>(a); \
    const double rhs = static_cast<double>(b); \
    if (std::abs(lhs - rhs) > static_cast<double>(tolerance)) { \
        throw std::runtime_error(std::string("near check failed: ") + #a + " vs " + #b); \
    } \
} while (false)

template <typename Exception, typename Function>
void checkThrows(Function&& function) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    CHECK(caught);
}

void run(const std::string& name, const std::function<void()>& test) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
}

vdb::RecallCommitConfig enabledConfig(vdb::RecallPolicy policy,
                                      double epsilon,
                                      size_t recall_k = 10) {
    vdb::RecallCommitConfig config;
    config.enabled = true;
    config.default_ack_mode = vdb::AckMode::Weak;
    config.policy = policy;
    config.epsilon = epsilon;
    config.k_min = recall_k;
    return config;
}

void testConfigValidation() {
    vdb::RecallCommitPolicyEvaluator defaults;
    const auto config = defaults.config();
    CHECK(!config.enabled);
    CHECK(config.default_ack_mode == vdb::AckMode::Stable);

    auto invalid = config;
    invalid.epsilon = 1.0;
    checkThrows<std::invalid_argument>([&] { vdb::RecallCommitPolicyEvaluator policy(invalid); });
    invalid = config;
    invalid.epsilon = std::numeric_limits<double>::quiet_NaN();
    checkThrows<std::invalid_argument>([&] { vdb::RecallCommitPolicyEvaluator policy(invalid); });
    invalid = config;
    invalid.k_min = 0;
    checkThrows<std::invalid_argument>([&] { vdb::RecallCommitPolicyEvaluator policy(invalid); });
}

void testStableAndDisabledRequests() {
    vdb::RecallCommitPolicyEvaluator disabled;
    auto decision = disabled.evaluateAdmission(vdb::AckMode::Weak, {});
    CHECK(decision.action == vdb::AdmissionAction::UseStable);
    CHECK(decision.reason == vdb::AdmissionReason::WeakAckDisabled);

    auto config = enabledConfig(vdb::RecallPolicy::Strict, 0.5);
    vdb::RecallCommitPolicyEvaluator policy(config);
    decision = policy.evaluateAdmission(vdb::AckMode::Stable, {});
    CHECK(decision.action == vdb::AdmissionAction::UseStable);
    CHECK(decision.reason == vdb::AdmissionReason::StableRequested);

    const auto counters = policy.counters();
    CHECK(counters.stable_requested == 1);
}

void testStrictBoundaryAndRetry() {
    auto config = enabledConfig(vdb::RecallPolicy::Strict, 0.2, 10);
    vdb::RecallCommitPolicyEvaluator policy(config);
    CHECK(vdb::RecallCommitPolicyEvaluator::policyRecordCap(config, 1000) == 2);

    auto decision = policy.evaluateAdmission(vdb::AckMode::Weak, {}, 2);
    CHECK(decision.action == vdb::AdmissionAction::AdmitWeak);
    CHECK(decision.candidate_weak_records == 2);
    CHECK_NEAR(decision.estimated_recall_loss, 0.2, 1e-12);

    decision = policy.evaluateAdmission(vdb::AckMode::Weak, {}, 3);
    CHECK(decision.action == vdb::AdmissionAction::UseStable);
    CHECK(decision.reason == vdb::AdmissionReason::StrictRecallLimit);

    decision = policy.evaluateAdmission(
        vdb::AckMode::Weak, vdb::AdmissionState{100, 1, 64}, 2, 128);
    CHECK(decision.action == vdb::AdmissionAction::FenceThenRetry);
    CHECK(decision.reason == vdb::AdmissionReason::StrictRecallLimit);
}

void testStrictZeroWindowDoesNotForceOne() {
    auto config = enabledConfig(vdb::RecallPolicy::Strict, 0.05, 10);
    vdb::RecallCommitPolicyEvaluator policy(config);
    CHECK(vdb::RecallCommitPolicyEvaluator::policyRecordCap(config, 1'000'000) == 0);
    const auto decision = policy.evaluateAdmission(vdb::AckMode::Weak, {}, 1);
    CHECK(decision.action == vdb::AdmissionAction::UseStable);
}

void testExchangeableBoundary() {
    auto config = enabledConfig(vdb::RecallPolicy::ExchangeableMean, 0.1);
    vdb::RecallCommitPolicyEvaluator policy(config);
    CHECK(vdb::RecallCommitPolicyEvaluator::policyRecordCap(config, 90) == 10);

    auto decision = policy.evaluateAdmission(
        vdb::AckMode::Weak, vdb::AdmissionState{90, 0, 0}, 10);
    CHECK(decision.action == vdb::AdmissionAction::AdmitWeak);
    CHECK_NEAR(decision.estimated_recall_loss, 0.1, 1e-12);

    decision = policy.evaluateAdmission(
        vdb::AckMode::Weak, vdb::AdmissionState{90, 10, 100}, 1);
    CHECK(decision.action == vdb::AdmissionAction::FenceThenRetry);
    CHECK(decision.reason == vdb::AdmissionReason::ExchangeableRecallLimit);

    decision = policy.evaluateAdmission(vdb::AckMode::Weak, {}, 1);
    CHECK(decision.action == vdb::AdmissionAction::UseStable);
}

void testOperationalCapsAndOverflow() {
    auto config = enabledConfig(vdb::RecallPolicy::Strict, 0.9, 10);
    config.max_tail_records = 2;
    config.max_tail_bytes = 100;
    vdb::RecallCommitPolicyEvaluator policy(config);

    auto decision = policy.evaluateAdmission(vdb::AckMode::Weak, {}, 3, 1);
    CHECK(decision.reason == vdb::AdmissionReason::RecordLimit);
    decision = policy.evaluateAdmission(
        vdb::AckMode::Weak, vdb::AdmissionState{100, 1, 90}, 1, 11);
    CHECK(decision.action == vdb::AdmissionAction::FenceThenRetry);
    CHECK(decision.reason == vdb::AdmissionReason::ByteLimit);
    decision = policy.evaluateAdmission(
        vdb::AckMode::Weak,
        vdb::AdmissionState{100, 1, std::numeric_limits<size_t>::max()},
        1,
        1);
    CHECK(decision.reason == vdb::AdmissionReason::ByteLimit);
    decision = policy.evaluateAdmission(
        vdb::AckMode::Weak,
        vdb::AdmissionState{100, std::numeric_limits<size_t>::max(), 0},
        1,
        0);
    CHECK(decision.reason == vdb::AdmissionReason::RecordLimit);

    CHECK_NEAR(vdb::RecallCommitPolicyEvaluator::expectedRecallLoss(
                   std::numeric_limits<size_t>::max(),
                   std::numeric_limits<size_t>::max()),
               0.5,
               1e-12);

    checkThrows<std::invalid_argument>(
        [&] { (void)policy.evaluateAdmission(vdb::AckMode::Weak, {}, 0); });
}

void testHypergeometricTail() {
    const double impossible =
        vdb::RecallCommitPolicyEvaluator::hypergeometricViolationProbability(100, 10, 10, 10);
    CHECK_NEAR(impossible, 0.0, 0.0);
    const double any_hit =
        vdb::RecallCommitPolicyEvaluator::hypergeometricViolationProbability(100, 10, 10, 0);
    CHECK(any_hit > 0.6 && any_hit < 0.7);

    auto config = enabledConfig(vdb::RecallPolicy::HypergeometricTail, 0.2, 10);
    config.delta = 0.05;
    vdb::RecallCommitPolicyEvaluator policy(config);
    auto decision = policy.evaluateAdmission(
        vdb::AckMode::Weak, vdb::AdmissionState{990, 0, 0}, 10);
    CHECK(decision.action == vdb::AdmissionAction::AdmitWeak);
    decision = policy.evaluateAdmission(
        vdb::AckMode::Weak, vdb::AdmissionState{500, 0, 0}, 500);
    CHECK(decision.action == vdb::AdmissionAction::UseStable);
    CHECK(decision.reason == vdb::AdmissionReason::HypergeometricTailLimit);
}

void testCorrelationGuardAndFallback() {
    auto config = enabledConfig(vdb::RecallPolicy::ExchangeableMean, 0.2);
    config.correlation_guard_enabled = true;
    config.correlation_min_queries = 2;
    config.correlation_min_expected_hits = 1.0;
    config.correlation_enrichment_threshold = 2.0;
    config.correlation_cusum_drift = 0.0;
    config.correlation_cusum_threshold = 100.0;
    vdb::RecallCommitPolicyEvaluator policy(config);

    policy.observeQuery(100, 10, 10, 5);
    policy.observeQuery(100, 10, 10, 5);
    auto correlation = policy.correlationCounters();
    CHECK(correlation.alarmed);
    CHECK(correlation.alarm_count == 1);
    CHECK(correlation.volatile_result_hits == 10);
    CHECK_NEAR(correlation.expected_volatile_hits, 2.0, 1e-12);
    CHECK(correlation.enrichment > 3.0);

    auto decision = policy.evaluateAdmission(
        vdb::AckMode::Weak, vdb::AdmissionState{100, 2, 64});
    CHECK(decision.action == vdb::AdmissionAction::FenceThenRetry);
    CHECK(decision.reason == vdb::AdmissionReason::CorrelationAlarm);
    decision = policy.evaluateAdmission(vdb::AckMode::Weak, vdb::AdmissionState{102, 0, 0});
    CHECK(decision.action == vdb::AdmissionAction::UseStable);

    policy.clearCorrelationAlarm();
    correlation = policy.correlationCounters();
    CHECK(!correlation.alarmed);
    CHECK(correlation.correlation_queries == 0);
    CHECK(correlation.alarm_count == 1);
}

void testCorrelationNullDoesNotAlarm() {
    auto config = enabledConfig(vdb::RecallPolicy::ExchangeableMean, 0.2);
    config.correlation_guard_enabled = true;
    config.correlation_min_queries = 10;
    config.correlation_min_expected_hits = 5.0;
    config.correlation_enrichment_threshold = 1.5;
    config.correlation_cusum_drift = 0.25;
    config.correlation_cusum_threshold = 8.0;
    vdb::RecallCommitPolicyEvaluator policy(config);

    for (size_t i = 0; i < 100; ++i) {
        policy.observeQuery(100, 10, 10, 1);  // Exactly the null expectation.
    }
    const auto correlation = policy.correlationCounters();
    CHECK(!correlation.alarmed);
    CHECK_NEAR(correlation.enrichment, 1.0, 1e-12);
    CHECK_NEAR(correlation.cusum, 0.0, 1e-12);
}

void testCorrelationInputValidation() {
    auto config = enabledConfig(vdb::RecallPolicy::ExchangeableMean, 0.2);
    vdb::RecallCommitPolicyEvaluator policy(config);
    checkThrows<std::invalid_argument>([&] { policy.observeQuery(10, 11, 5, 1); });
    checkThrows<std::invalid_argument>([&] { policy.observeQuery(10, 2, 5, 3); });
    checkThrows<std::invalid_argument>([&] { policy.observeQuery(10, 2, 1, 2); });
}

void testDurabilityFrontier() {
    vdb::DurabilityFrontier frontier(7);
    auto snapshot = frontier.snapshot();
    CHECK(snapshot.visible_lsn == 7);
    CHECK(snapshot.durable_lsn == 7);
    CHECK(frontier.isDurable(7));
    CHECK(!frontier.isDurable(8));

    frontier.publishVisible(9);  // A failed reservation may leave an LSN gap.
    snapshot = frontier.snapshot();
    CHECK(snapshot.visible_lsn == 9);
    CHECK(snapshot.durable_lsn == 7);
    frontier.advanceDurable(9);
    CHECK(frontier.isDurable(8));

    checkThrows<std::logic_error>([&] { frontier.publishVisible(9); });
    checkThrows<std::logic_error>([&] { frontier.advanceDurable(8); });
    frontier.publishVisible(12);
    checkThrows<std::logic_error>([&] { frontier.advanceDurable(13); });

    frontier.resetRecovered(20);
    snapshot = frontier.snapshot();
    CHECK(snapshot.visible_lsn == 20 && snapshot.durable_lsn == 20);
}

void testConcurrentPolicyCounters() {
    auto config = enabledConfig(vdb::RecallPolicy::Strict, 0.9, 10);
    vdb::RecallCommitPolicyEvaluator policy(config);
    constexpr size_t thread_count = 8;
    constexpr size_t iterations = 200;
    std::vector<std::thread> threads;
    std::atomic<bool> decisions_correct{true};
    threads.reserve(thread_count);
    for (size_t thread = 0; thread < thread_count; ++thread) {
        threads.emplace_back([&] {
            for (size_t i = 0; i < iterations; ++i) {
                const auto decision = policy.evaluateAdmission(
                    vdb::AckMode::Weak, vdb::AdmissionState{100, 0, 0});
                if (decision.action != vdb::AdmissionAction::AdmitWeak) {
                    decisions_correct.store(false, std::memory_order_relaxed);
                }
                policy.observeQuery(100, 0, 10, 0);
            }
        });
    }
    for (auto& thread : threads) thread.join();

    CHECK(decisions_correct.load(std::memory_order_relaxed));
    const auto counters = policy.counters();
    CHECK(counters.admission_checks == thread_count * iterations);
    CHECK(counters.weak_admitted == thread_count * iterations);
    CHECK(counters.correlation.correlation_queries == thread_count * iterations);
}

void testReceiptAndStatusAreIntegrationReady() {
    vdb::CommitReceipt receipt;
    CHECK(!receipt.applied);
    CHECK(receipt.actual_ack == vdb::AckLevel::None);
    CHECK(!receipt.provisional);

    vdb::DurabilityStatus status;
    CHECK(status.visible_lsn == 0);
    CHECK(status.durable_lsn == 0);
    CHECK(status.health == vdb::CommitterHealth::Healthy);
}

std::filesystem::path tempDb(const std::string& name) {
    return std::filesystem::temp_directory_path() /
           ("vdb_committer_" + name + "_" + std::to_string(::getpid()));
}

vdb::RecallCommitConfig strictConfig(double epsilon = 0.2) {
    auto config = enabledConfig(vdb::RecallPolicy::Strict, epsilon, 10);
    config.max_tail_records = 64;
    config.max_tail_bytes = 1u << 20;
    return config;
}

void testRealWeakVisibilityAndFence() {
    const auto path = tempDb("visibility");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                      false, 0, path.string());
    db.configureRecallCommit(strictConfig());
    db.initialize();

    const auto receipt = db.insertWithAck(
        Vector(std::vector<float>{0.0f, 0.0f}), "weak", "payload", vdb::AckMode::Weak);
    CHECK(receipt.applied);
    CHECK(receipt.actual_ack == vdb::AckLevel::Weak);
    CHECK(receipt.provisional);
    CHECK(receipt.visible_lsn > receipt.durable_lsn);

    const Vector query(std::vector<float>{0.0f, 0.0f});
    const auto latest = db.similaritySearch(query, 1, vdb::ReadVisibility::Latest);
    const auto stable = db.similaritySearch(query, 1, vdb::ReadVisibility::Stable);
    CHECK(latest.results.size() == 1 && latest.results[0].key == "weak");
    CHECK(stable.results.empty());
    CHECK(db.durabilityStatus().weak_records == 1);

    CHECK(db.durabilityFence() >= receipt.lsn);
    CHECK(db.waitUntilDurable(receipt.lsn, std::chrono::seconds(1)));
    CHECK(db.durabilityStatus().weak_records == 0);
    CHECK(db.similaritySearch(query, 1, vdb::ReadVisibility::Stable).results[0].key == "weak");
    db.shutdown();
    std::filesystem::remove_all(path);
}

void testFenceDoesNotTriggerIndexMaintenance() {
    const auto path = tempDb("fence_representation");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                      false, 0, path.string());
    db.configureSegmentedStorage(1, 16, 0.25);
    db.configureRecallCommit(strictConfig());
    db.initialize();

    const Vector value(std::vector<float>{0.25f, 0.5f});
    const auto weak = db.insertWithAck(value, "weak", vdb::AckMode::Weak);
    CHECK(weak.actual_ack == vdb::AckLevel::Weak);
    CHECK(db.getStatistics().segmented_stats.sealed_segments == 0);

    CHECK(db.durabilityFence() >= weak.lsn);
    const auto after_fence = db.getStatistics().segmented_stats;
    CHECK(after_fence.sealed_segments == 0);
    CHECK(after_fence.mutable_records == 1);
    const auto stable = db.similaritySearch(value, 1, vdb::ReadVisibility::Stable);
    CHECK(stable.results.size() == 1 && stable.results[0].key == "weak");

    db.sealMutableSegment();
    CHECK(db.getStatistics().segmented_stats.sealed_segments == 1);
    db.shutdown();
    std::filesystem::remove_all(path);
}

void testReadOnlyRecoveryRejectsMutations() {
    const auto path = tempDb("read_only");
    std::filesystem::remove_all(path);
    {
        VectorDatabase writer(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                              false, 0, path.string());
        writer.initialize();
        CHECK(writer.insert(Vector(std::vector<float>{1.0f, 2.0f}), "stable"));
        writer.shutdown();
    }

    {
        VectorDatabase reader(2, VectorDatabase::SearchMode::HNSW, false, true, {},
                              false, 0, path.string(),
                              VectorDatabase::StorageEngine::Segmented,
                              vdb::OpenMode::ReadOnlyRecovery);
        reader.initialize();
        const Vector value(std::vector<float>{2.0f, 3.0f});
        CHECK(reader.get("stable").has_value());
        checkThrows<std::logic_error>([&] {
            (void)reader.insertWithAck(value, "new", vdb::AckMode::Stable);
        });
        checkThrows<std::logic_error>([&] { (void)reader.update(value, "stable"); });
        checkThrows<std::logic_error>([&] { (void)reader.remove("stable"); });
        checkThrows<std::logic_error>([&] { (void)reader.durabilityFence(); });
        checkThrows<std::logic_error>([&] { reader.sealMutableSegment(); });
        checkThrows<std::logic_error>([&] { reader.compactSegments(); });
        checkThrows<std::logic_error>([&] { (void)reader.flush(); });
        checkThrows<std::logic_error>([&] { (void)reader.checkpoint(); });
        checkThrows<std::logic_error>([&] {
            reader.setDistanceMetric(std::make_shared<ManhattanDistance>());
        });
        reader.shutdown();
    }

    {
        VectorDatabase recovered(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                                 false, 0, path.string());
        recovered.initialize();
        CHECK(recovered.get("stable").has_value());
        CHECK(!recovered.get("new").has_value());
        recovered.shutdown();
    }
    std::filesystem::remove_all(path);
}

void testInvalidPayloadDoesNotPoisonCommitter() {
    const auto path = tempDb("invalid_payload");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, true, {},
                      false, 0, path.string());
    db.configureRecallCommit(strictConfig());
    db.initialize();

    const Vector value(std::vector<float>{1.0f, 2.0f});
    const std::string oversized_key((1u << 20) + 1, 'k');
    checkThrows<std::length_error>([&] {
        (void)db.insertWithAck(value, oversized_key, vdb::AckMode::Weak);
    });
    CHECK(db.durabilityStatus().health == vdb::CommitterHealth::Healthy);

    const auto valid = db.insertWithAck(value, "valid", vdb::AckMode::Weak);
    CHECK(valid.applied);
    CHECK(valid.actual_ack == vdb::AckLevel::Weak);
    (void)db.durabilityFence();

    const auto batch = db.batchInsert(
        {"batch-valid", oversized_key}, {value, value});
    CHECK(!batch.success);
    CHECK(batch.operations_committed == 1);
    CHECK(db.durabilityStatus().weak_records == 0);
    CHECK(db.get("batch-valid").has_value());
    db.shutdown();
    std::filesystem::remove_all(path);
}

void testAgeFencePreemptsGroupDelay() {
    const auto path = tempDb("age_preempts_group");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                      false, 0, path.string());
    auto config = strictConfig();
    config.group_delay = std::chrono::milliseconds(500);
    config.max_tail_age = std::chrono::milliseconds(20);
    db.configureRecallCommit(config);
    db.initialize();

    const auto weak = db.insertWithAck(
        Vector(std::vector<float>{3.0f, 4.0f}), "weak", vdb::AckMode::Weak);
    CHECK(weak.actual_ack == vdb::AckLevel::Weak);
    CHECK(db.waitUntilDurable(weak.lsn, std::chrono::milliseconds(250)));
    CHECK(db.recallCommitterStatistics().age_fences >= 1);
    db.shutdown();
    std::filesystem::remove_all(path);
}

void testStrictZeroCapAutoStable() {
    const auto path = tempDb("zero_cap");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                      false, 0, path.string());
    db.configureRecallCommit(strictConfig(0.05));
    db.initialize();
    const auto receipt = db.insertWithAck(
        Vector(std::vector<float>{1.0f, 2.0f}), "stable", vdb::AckMode::Weak);
    CHECK(receipt.applied);
    CHECK(receipt.actual_ack == vdb::AckLevel::Stable);
    CHECK(!receipt.provisional);
    CHECK(receipt.visible_lsn == receipt.durable_lsn);
    CHECK(db.durabilityStatus().weak_records == 0);
    db.shutdown();
    std::filesystem::remove_all(path);
}

void testExchangeableAdmissionOnRealStore() {
    const auto path = tempDb("exchangeable");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                      false, 0, path.string());
    auto config = enabledConfig(vdb::RecallPolicy::ExchangeableMean, 0.2, 10);
    config.max_tail_records = 64;
    db.configureRecallCommit(config);
    db.initialize();

    for (size_t i = 0; i < 4; ++i) {
        const auto receipt = db.insertWithAck(
            Vector(std::vector<float>{static_cast<float>(i), 0.0f}),
            "d" + std::to_string(i), vdb::AckMode::Stable);
        CHECK(receipt.actual_ack == vdb::AckLevel::Stable);
    }
    const auto weak = db.insertWithAck(
        Vector(std::vector<float>{9.0f, 0.0f}), "w", vdb::AckMode::Weak);
    CHECK(weak.actual_ack == vdb::AckLevel::Weak);
    CHECK(weak.risk_estimate <= 0.2 + 1e-12);
    CHECK(db.durabilityStatus().weak_records == 1);
    db.shutdown();
    std::filesystem::remove_all(path);
}

void testConcurrentAdmissionNeverOvershoots() {
    const auto path = tempDb("concurrent_cap");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                      false, 0, path.string());
    auto config = strictConfig(0.2);  // floor(epsilon*k)=2
    config.group_delay = std::chrono::microseconds(500);
    db.configureRecallCommit(config);
    db.initialize();

    constexpr size_t writers = 16;
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<vdb::WriteReceipt> receipts(writers);
    for (size_t i = 0; i < writers; ++i) {
        threads.emplace_back([&, i] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            receipts[i] = db.insertWithAck(
                Vector(std::vector<float>{static_cast<float>(i), 0.0f}),
                "k" + std::to_string(i), vdb::AckMode::Weak);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
    for (const auto& receipt : receipts) CHECK(receipt.applied);
    CHECK(db.durabilityStatus().weak_records <= 2);
    CHECK(db.recallCommitterStatistics().max_weak_records <= 2);
    CHECK(db.recallPolicyStatistics().cap_overshoots == 0);
    (void)db.durabilityFence();
    CHECK(db.vectorCount() == writers);
    db.shutdown();
    std::filesystem::remove_all(path);
}

void testStableFollowersGroupAndAgeFence() {
    const auto path = tempDb("group_age");
    std::filesystem::remove_all(path);
    VectorDatabase db(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                      false, 0, path.string());
    auto config = strictConfig(0.2);
    config.group_delay = std::chrono::milliseconds(5);
    config.max_tail_age = std::chrono::milliseconds(20);
    db.configureRecallCommit(config);
    db.initialize();

    constexpr size_t writers = 12;
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    std::vector<vdb::AckLevel> levels(writers, vdb::AckLevel::None);
    for (size_t i = 0; i < writers; ++i) {
        threads.emplace_back([&, i] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            auto receipt = db.insertWithAck(
                Vector(std::vector<float>{static_cast<float>(i), 1.0f}),
                "s" + std::to_string(i), vdb::AckMode::Stable);
            levels[i] = receipt.actual_ack;
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
    for (const auto level : levels) CHECK(level == vdb::AckLevel::Stable);
    auto stats = db.recallCommitterStatistics();
    CHECK(stats.sync_successes < writers);
    CHECK(stats.follower_requests > 0);

    const auto weak = db.insertWithAck(
        Vector(std::vector<float>{99.0f, 1.0f}), "age", vdb::AckMode::Weak);
    CHECK(weak.actual_ack == vdb::AckLevel::Weak);
    CHECK(db.waitUntilDurable(weak.lsn, std::chrono::seconds(2)));
    CHECK(db.recallCommitterStatistics().age_fences >= 1);
    db.shutdown();
    std::filesystem::remove_all(path);
}

}  // namespace

int main() {
    run("config validation", testConfigValidation);
    run("stable and disabled requests", testStableAndDisabledRequests);
    run("strict boundary and retry", testStrictBoundaryAndRetry);
    run("strict zero window", testStrictZeroWindowDoesNotForceOne);
    run("exchangeable boundary", testExchangeableBoundary);
    run("operational caps and overflow", testOperationalCapsAndOverflow);
    run("hypergeometric tail", testHypergeometricTail);
    run("correlation guard and fallback", testCorrelationGuardAndFallback);
    run("correlation null workload", testCorrelationNullDoesNotAlarm);
    run("correlation input validation", testCorrelationInputValidation);
    run("durability frontier", testDurabilityFrontier);
    run("concurrent policy counters", testConcurrentPolicyCounters);
    run("receipt and status", testReceiptAndStatusAreIntegrationReady);
    run("real weak visibility and fence", testRealWeakVisibilityAndFence);
    run("fence preserves exact representation", testFenceDoesNotTriggerIndexMaintenance);
    run("read-only recovery rejects mutations", testReadOnlyRecoveryRejectsMutations);
    run("invalid payload keeps committer healthy", testInvalidPayloadDoesNotPoisonCommitter);
    run("age fence preempts group delay", testAgeFencePreemptsGroupDelay);
    run("strict zero cap auto-stable", testStrictZeroCapAutoStable);
    run("real exchangeable admission", testExchangeableAdmissionOnRealStore);
    run("concurrent admission cap", testConcurrentAdmissionNeverOvershoots);
    run("stable grouping and age fence", testStableFollowersGroupAndAgeFence);

    if (failures != 0) {
        std::cerr << failures << " committer policy test(s) failed\n";
        return 1;
    }
    std::cout << "All committer policy tests passed\n";
    return 0;
}
