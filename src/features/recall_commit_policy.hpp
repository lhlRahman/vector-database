#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

namespace vdb {

// AckMode is the caller's request. AckLevel is what the database actually
// returned; a weak request can be promoted to Stable when no weak window is
// admissible.
enum class AckMode {
    Stable,
    Weak,
    WeakAllowed = Weak,
};

enum class AckLevel {
    None,
    Weak,
    Stable,
};

enum class RecallPolicy {
    Strict,
    ExchangeableMean,
    HypergeometricTail,
};

enum class ReadVisibility {
    Latest,
    Stable,
    IncludeWeak = Latest,
    DurableOnly = Stable,
};

enum class OpenMode {
    ReadWrite,
    ReadOnlyRecovery,
};

enum class CommitterHealth {
    Healthy,
    SyncFailed,
    ShuttingDown,
};

struct RecallCommitConfig {
    bool enabled{false};
    AckMode default_ack_mode{AckMode::Stable};
    RecallPolicy policy{RecallPolicy::Strict};
    ReadVisibility read_visibility{ReadVisibility::Latest};

    // Target recall loss at k_min. epsilon must be finite and in [0, 1).
    // delta is the allowed exceedance probability for HypergeometricTail.
    double epsilon{0.0};
    double delta{0.01};
    size_t k_min{10};

    // Independent operational caps. A zero age/delay disables that timer.
    size_t max_tail_records{std::numeric_limits<size_t>::max()};
    size_t max_tail_bytes{std::numeric_limits<size_t>::max()};
    std::chrono::milliseconds max_tail_age{0};
    std::chrono::microseconds group_delay{0};
    uint32_t hnsw_seed{100};

    // Cheap online correlation guard. For each query, compare observed weak
    // result hits with the exchangeable null k*W/N and accumulate a one-sided
    // normalized CUSUM. The alarm is latched until explicitly cleared.
    bool correlation_guard_enabled{false};
    size_t correlation_min_queries{32};
    double correlation_min_expected_hits{5.0};
    double correlation_prior_hits{1.0};
    double correlation_enrichment_threshold{2.0};
    double correlation_cusum_drift{0.25};
    double correlation_cusum_threshold{8.0};
};

struct AdmissionState {
    size_t durable_records{0};
    size_t weak_records{0};
    size_t weak_bytes{0};
};

enum class AdmissionAction {
    AdmitWeak,
    FenceThenRetry,
    UseStable,
};

enum class AdmissionReason {
    Allowed,
    StableRequested,
    WeakAckDisabled,
    StrictRecallLimit,
    ExchangeableRecallLimit,
    HypergeometricTailLimit,
    RecordLimit,
    ByteLimit,
    CorrelationAlarm,
};

struct AdmissionDecision {
    AdmissionAction action{AdmissionAction::UseStable};
    AdmissionReason reason{AdmissionReason::WeakAckDisabled};
    size_t candidate_weak_records{0};
    size_t policy_record_cap{0};
    double estimated_recall_loss{0.0};
    double estimated_violation_probability{0.0};
    bool correlation_alarm{false};
};

struct CorrelationGuardCounters {
    uint64_t correlation_queries{0};
    uint64_t result_slots{0};
    uint64_t volatile_result_hits{0};
    double expected_volatile_hits{0.0};
    double expected_variance{0.0};
    double enrichment{1.0};
    double cusum{0.0};
    uint64_t alarm_count{0};
    bool alarmed{false};
};

struct RecallCommitPolicyCounters {
    uint64_t admission_checks{0};
    uint64_t weak_admitted{0};
    uint64_t stable_requested{0};
    uint64_t auto_stable{0};
    uint64_t fence_then_retry{0};
    uint64_t strict_rejections{0};
    uint64_t exchangeable_rejections{0};
    uint64_t hypergeometric_rejections{0};
    uint64_t record_limit_rejections{0};
    uint64_t byte_limit_rejections{0};
    uint64_t correlation_rejections{0};
    uint64_t cap_overshoots{0};
    CorrelationGuardCounters correlation{};
};

struct DurabilityFrontierSnapshot {
    uint64_t visible_lsn{0};
    uint64_t durable_lsn{0};
};

struct DurabilityStatus {
    uint64_t appended_lsn{0};
    uint64_t visible_lsn{0};
    uint64_t durable_lsn{0};
    size_t visible_records{0};
    size_t durable_records{0};
    size_t weak_records{0};
    size_t weak_bytes{0};
    size_t policy_record_cap{0};
    double estimated_recall_loss{0.0};
    RecallPolicy configured_policy{RecallPolicy::Strict};
    RecallPolicy effective_policy{RecallPolicy::Strict};
    bool correlation_alarm{false};
    uint64_t manifest_generation{0};
    CommitterHealth health{CommitterHealth::Healthy};
};

struct CommitReceipt {
    bool applied{false};
    uint64_t lsn{0};
    AckMode requested_ack{AckMode::Stable};
    AckLevel actual_ack{AckLevel::None};
    bool provisional{false};
    uint64_t visible_lsn{0};
    uint64_t durable_lsn{0};
    size_t durable_count{0};
    size_t weak_count{0};
    size_t policy_cap{0};
    double risk_estimate{0.0};
    bool correlation_alarm{false};
};

using WriteReceipt = CommitReceipt;

class DurabilityFrontier {
public:
    explicit DurabilityFrontier(uint64_t recovered_lsn = 0);

    // Recovery establishes a new process-local baseline: every record that the
    // production recovery path accepted is durable on the recovered image.
    void resetRecovered(uint64_t recovered_lsn);

    // Gaps are allowed (a reserved LSN may fail before publication), but neither
    // frontier may regress and durable may never pass visible.
    void publishVisible(uint64_t lsn);
    void advanceDurable(uint64_t lsn);

    [[nodiscard]] DurabilityFrontierSnapshot snapshot() const;
    [[nodiscard]] bool isDurable(uint64_t lsn) const;

private:
    mutable std::mutex mutex_;
    uint64_t visible_lsn_{0};
    uint64_t durable_lsn_{0};
};

class RecallCommitPolicyEvaluator {
public:
    explicit RecallCommitPolicyEvaluator(RecallCommitConfig config = {});

    void updateConfig(const RecallCommitConfig& config);
    [[nodiscard]] RecallCommitConfig config() const;

    // This method only decides; it performs no I/O and changes no frontier.
    // A caller receiving FenceThenRetry must harden its current weak window and
    // call evaluateAdmission again with weak_records == 0.
    [[nodiscard]] AdmissionDecision evaluateAdmission(
        AckMode requested_mode,
        const AdmissionState& state,
        size_t incoming_records = 1,
        size_t incoming_bytes = 0);

    // Record how many of a query's returned slots came from the current weak
    // window. Inputs describe the state at the query's linearization point.
    void observeQuery(size_t visible_records,
                      size_t weak_records,
                      size_t result_slots,
                      size_t volatile_result_hits);

    // Clears the latched alarm and its current statistical window while keeping
    // the lifetime alarm_count and admission counters.
    void clearCorrelationAlarm();

    [[nodiscard]] CorrelationGuardCounters correlationCounters() const;
    [[nodiscard]] RecallCommitPolicyCounters counters() const;

    static void validateConfig(const RecallCommitConfig& config);
    [[nodiscard]] static size_t policyRecordCap(const RecallCommitConfig& config,
                                                size_t durable_records);
    [[nodiscard]] static double expectedRecallLoss(size_t durable_records,
                                                   size_t weak_records);
    [[nodiscard]] static double hypergeometricViolationProbability(
        size_t population,
        size_t weak_records,
        size_t draws,
        size_t allowed_weak_hits);

private:
    static size_t saturatingAdd(size_t a, size_t b);
    static AdmissionAction rejectionAction(const AdmissionState& state);

    AdmissionDecision evaluateAdmissionLocked(AckMode requested_mode,
                                                const AdmissionState& state,
                                                size_t incoming_records,
                                                size_t incoming_bytes);
    void updateCorrelationDerivedLocked();

    mutable std::mutex mutex_;
    RecallCommitConfig config_;
    RecallCommitPolicyCounters counters_{};
};

}  // namespace vdb
