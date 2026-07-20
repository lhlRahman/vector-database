#include "recall_commit_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vdb {
namespace {

long double logChoose(size_t n, size_t k) {
    if (k > n) return -INFINITY;
    k = std::min(k, n - k);
    return std::lgammal(static_cast<long double>(n) + 1.0L) -
           std::lgammal(static_cast<long double>(k) + 1.0L) -
           std::lgammal(static_cast<long double>(n - k) + 1.0L);
}

size_t floorToSize(long double value) {
    if (!(value > 0.0L)) return 0;
    const long double max_size = static_cast<long double>(std::numeric_limits<size_t>::max());
    if (value >= max_size) return std::numeric_limits<size_t>::max();
    return static_cast<size_t>(std::floor(value));
}

}  // namespace

DurabilityFrontier::DurabilityFrontier(uint64_t recovered_lsn)
    : visible_lsn_(recovered_lsn), durable_lsn_(recovered_lsn) {}

void DurabilityFrontier::resetRecovered(uint64_t recovered_lsn) {
    std::lock_guard lock(mutex_);
    visible_lsn_ = recovered_lsn;
    durable_lsn_ = recovered_lsn;
}

void DurabilityFrontier::publishVisible(uint64_t lsn) {
    std::lock_guard lock(mutex_);
    if (lsn == 0 || lsn <= visible_lsn_) {
        throw std::logic_error("visible LSN must increase");
    }
    visible_lsn_ = lsn;
}

void DurabilityFrontier::advanceDurable(uint64_t lsn) {
    std::lock_guard lock(mutex_);
    if (lsn < durable_lsn_) {
        throw std::logic_error("durable LSN must not regress");
    }
    if (lsn > visible_lsn_) {
        throw std::logic_error("durable LSN must not pass visible LSN");
    }
    durable_lsn_ = lsn;
}

DurabilityFrontierSnapshot DurabilityFrontier::snapshot() const {
    std::lock_guard lock(mutex_);
    return DurabilityFrontierSnapshot{visible_lsn_, durable_lsn_};
}

bool DurabilityFrontier::isDurable(uint64_t lsn) const {
    std::lock_guard lock(mutex_);
    return lsn != 0 && lsn <= durable_lsn_;
}

RecallCommitPolicyEvaluator::RecallCommitPolicyEvaluator(RecallCommitConfig config)
    : config_(std::move(config)) {
    validateConfig(config_);
}

void RecallCommitPolicyEvaluator::updateConfig(const RecallCommitConfig& config) {
    validateConfig(config);
    std::lock_guard lock(mutex_);
    config_ = config;
    updateCorrelationDerivedLocked();
}

RecallCommitConfig RecallCommitPolicyEvaluator::config() const {
    std::lock_guard lock(mutex_);
    return config_;
}

size_t RecallCommitPolicyEvaluator::saturatingAdd(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return std::numeric_limits<size_t>::max();
    }
    return a + b;
}

AdmissionAction RecallCommitPolicyEvaluator::rejectionAction(const AdmissionState& state) {
    return (state.weak_records != 0 || state.weak_bytes != 0)
               ? AdmissionAction::FenceThenRetry
               : AdmissionAction::UseStable;
}

AdmissionDecision RecallCommitPolicyEvaluator::evaluateAdmission(
    AckMode requested_mode,
    const AdmissionState& state,
    size_t incoming_records,
    size_t incoming_bytes) {
    if (incoming_records == 0) {
        throw std::invalid_argument("incoming_records must be nonzero");
    }
    std::lock_guard lock(mutex_);
    return evaluateAdmissionLocked(requested_mode, state, incoming_records, incoming_bytes);
}

AdmissionDecision RecallCommitPolicyEvaluator::evaluateAdmissionLocked(
    AckMode requested_mode,
    const AdmissionState& state,
    size_t incoming_records,
    size_t incoming_bytes) {
    ++counters_.admission_checks;

    AdmissionDecision decision;
    const bool record_count_overflow =
        incoming_records > std::numeric_limits<size_t>::max() - state.weak_records;
    const bool byte_count_overflow =
        incoming_bytes > std::numeric_limits<size_t>::max() - state.weak_bytes;
    decision.candidate_weak_records = saturatingAdd(state.weak_records, incoming_records);
    decision.policy_record_cap = policyRecordCap(config_, state.durable_records);
    if (config_.policy == RecallPolicy::Strict) {
        decision.estimated_recall_loss =
            static_cast<double>(std::min(decision.candidate_weak_records, config_.k_min)) /
            static_cast<double>(config_.k_min);
    } else {
        decision.estimated_recall_loss =
            expectedRecallLoss(state.durable_records, decision.candidate_weak_records);
    }
    decision.correlation_alarm = counters_.correlation.alarmed;

    if (requested_mode == AckMode::Stable) {
        ++counters_.stable_requested;
        decision.action = AdmissionAction::UseStable;
        decision.reason = AdmissionReason::StableRequested;
        return decision;
    }

    if (!config_.enabled) {
        ++counters_.auto_stable;
        decision.action = AdmissionAction::UseStable;
        decision.reason = AdmissionReason::WeakAckDisabled;
        return decision;
    }

    if (config_.correlation_guard_enabled && counters_.correlation.alarmed) {
        ++counters_.correlation_rejections;
        decision.action = rejectionAction(state);
        decision.reason = AdmissionReason::CorrelationAlarm;
        if (decision.action == AdmissionAction::FenceThenRetry) {
            ++counters_.fence_then_retry;
        } else {
            ++counters_.auto_stable;
        }
        return decision;
    }

    if (record_count_overflow || decision.candidate_weak_records > config_.max_tail_records) {
        ++counters_.record_limit_rejections;
        decision.action = rejectionAction(state);
        decision.reason = AdmissionReason::RecordLimit;
    } else if (byte_count_overflow ||
               saturatingAdd(state.weak_bytes, incoming_bytes) > config_.max_tail_bytes) {
        ++counters_.byte_limit_rejections;
        decision.action = rejectionAction(state);
        decision.reason = AdmissionReason::ByteLimit;
    } else {
        bool allowed = false;
        switch (config_.policy) {
            case RecallPolicy::Strict: {
                allowed = decision.candidate_weak_records <= decision.policy_record_cap;
                if (!allowed) {
                    ++counters_.strict_rejections;
                    decision.reason = AdmissionReason::StrictRecallLimit;
                }
                break;
            }
            case RecallPolicy::ExchangeableMean: {
                allowed = decision.estimated_recall_loss <= config_.epsilon;
                if (!allowed) {
                    ++counters_.exchangeable_rejections;
                    decision.reason = AdmissionReason::ExchangeableRecallLimit;
                }
                break;
            }
            case RecallPolicy::HypergeometricTail: {
                const size_t population =
                    saturatingAdd(state.durable_records, decision.candidate_weak_records);
                const size_t allowed_hits = floorToSize(
                    static_cast<long double>(config_.epsilon) * config_.k_min);
                decision.estimated_violation_probability =
                    hypergeometricViolationProbability(population,
                                                       decision.candidate_weak_records,
                                                       config_.k_min,
                                                       allowed_hits);
                allowed = decision.estimated_violation_probability <= config_.delta;
                if (!allowed) {
                    ++counters_.hypergeometric_rejections;
                    decision.reason = AdmissionReason::HypergeometricTailLimit;
                }
                break;
            }
        }

        if (allowed) {
            ++counters_.weak_admitted;
            decision.action = AdmissionAction::AdmitWeak;
            decision.reason = AdmissionReason::Allowed;
            return decision;
        }
        decision.action = rejectionAction(state);
    }

    if (state.weak_records > decision.policy_record_cap) {
        ++counters_.cap_overshoots;
    }
    if (decision.action == AdmissionAction::FenceThenRetry) {
        ++counters_.fence_then_retry;
    } else {
        ++counters_.auto_stable;
    }
    return decision;
}

void RecallCommitPolicyEvaluator::observeQuery(size_t visible_records,
                                               size_t weak_records,
                                               size_t result_slots,
                                               size_t volatile_result_hits) {
    if (weak_records > visible_records) {
        throw std::invalid_argument("weak_records exceeds visible_records");
    }
    if (volatile_result_hits > result_slots || volatile_result_hits > weak_records) {
        throw std::invalid_argument("volatile_result_hits exceeds possible hits");
    }

    std::lock_guard lock(mutex_);
    auto& correlation = counters_.correlation;
    ++correlation.correlation_queries;
    correlation.result_slots += result_slots;
    correlation.volatile_result_hits += volatile_result_hits;

    const size_t draws = std::min(result_slots, visible_records);
    double expected = 0.0;
    double variance = 0.0;
    if (visible_records != 0 && draws != 0 && weak_records != 0) {
        const double p = static_cast<double>(weak_records) /
                         static_cast<double>(visible_records);
        expected = static_cast<double>(draws) * p;
        variance = static_cast<double>(draws) * p * (1.0 - p);
        if (visible_records > 1) {
            variance *= static_cast<double>(visible_records - draws) /
                        static_cast<double>(visible_records - 1);
        } else {
            variance = 0.0;
        }
    }

    correlation.expected_volatile_hits += expected;
    correlation.expected_variance += variance;

    const double z = (static_cast<double>(volatile_result_hits) - expected) /
                     std::sqrt(std::max(variance, 1.0));
    correlation.cusum = std::max(
        0.0, correlation.cusum + z - config_.correlation_cusum_drift);
    updateCorrelationDerivedLocked();
}

void RecallCommitPolicyEvaluator::updateCorrelationDerivedLocked() {
    auto& correlation = counters_.correlation;
    correlation.enrichment =
        (static_cast<double>(correlation.volatile_result_hits) +
         config_.correlation_prior_hits) /
        (correlation.expected_volatile_hits + config_.correlation_prior_hits);

    if (!config_.correlation_guard_enabled || correlation.alarmed ||
        correlation.correlation_queries < config_.correlation_min_queries) {
        return;
    }

    const bool enrichment_alarm =
        correlation.expected_volatile_hits >= config_.correlation_min_expected_hits &&
        correlation.enrichment >= config_.correlation_enrichment_threshold;
    const bool cusum_alarm = correlation.cusum >= config_.correlation_cusum_threshold;
    if (enrichment_alarm || cusum_alarm) {
        correlation.alarmed = true;
        ++correlation.alarm_count;
    }
}

void RecallCommitPolicyEvaluator::clearCorrelationAlarm() {
    std::lock_guard lock(mutex_);
    const uint64_t alarm_count = counters_.correlation.alarm_count;
    counters_.correlation = CorrelationGuardCounters{};
    counters_.correlation.alarm_count = alarm_count;
}

CorrelationGuardCounters RecallCommitPolicyEvaluator::correlationCounters() const {
    std::lock_guard lock(mutex_);
    return counters_.correlation;
}

RecallCommitPolicyCounters RecallCommitPolicyEvaluator::counters() const {
    std::lock_guard lock(mutex_);
    return counters_;
}

void RecallCommitPolicyEvaluator::validateConfig(const RecallCommitConfig& config) {
    if (!std::isfinite(config.epsilon) || config.epsilon < 0.0 || config.epsilon >= 1.0) {
        throw std::invalid_argument("epsilon must be finite and in [0, 1)");
    }
    if (config.k_min == 0) {
        throw std::invalid_argument("k_min must be nonzero");
    }
    if (!std::isfinite(config.delta) || config.delta < 0.0 || config.delta > 1.0) {
        throw std::invalid_argument("delta must be in [0, 1]");
    }
    if (config.correlation_min_queries == 0) {
        throw std::invalid_argument("correlation_min_queries must be nonzero");
    }
    if (!std::isfinite(config.correlation_min_expected_hits) ||
        config.correlation_min_expected_hits < 0.0 ||
        !std::isfinite(config.correlation_prior_hits) || config.correlation_prior_hits <= 0.0 ||
        !std::isfinite(config.correlation_enrichment_threshold) ||
        config.correlation_enrichment_threshold < 1.0 ||
        !std::isfinite(config.correlation_cusum_drift) ||
        config.correlation_cusum_drift < 0.0 ||
        !std::isfinite(config.correlation_cusum_threshold) ||
        config.correlation_cusum_threshold <= 0.0) {
        throw std::invalid_argument("invalid correlation guard configuration");
    }
}

size_t RecallCommitPolicyEvaluator::policyRecordCap(const RecallCommitConfig& config,
                                                    size_t durable_records) {
    validateConfig(config);
    size_t cap = 0;
    switch (config.policy) {
        case RecallPolicy::Strict:
            cap = floorToSize(static_cast<long double>(config.epsilon) * config.k_min);
            break;
        case RecallPolicy::ExchangeableMean:
            cap = floorToSize(
                static_cast<long double>(config.epsilon) * durable_records /
                (1.0L - static_cast<long double>(config.epsilon)));
            break;
        case RecallPolicy::HypergeometricTail:
            // There is no useful closed-form cap. The evaluator computes the
            // exact tail probability for the proposed window; expose the hard
            // operational cap here for status reporting.
            cap = config.max_tail_records;
            break;
    }
    return std::min(cap, config.max_tail_records);
}

double RecallCommitPolicyEvaluator::expectedRecallLoss(size_t durable_records,
                                                       size_t weak_records) {
    if (durable_records == 0 && weak_records == 0) return 0.0;
    const long double population = static_cast<long double>(durable_records) +
                                   static_cast<long double>(weak_records);
    return static_cast<double>(static_cast<long double>(weak_records) / population);
}

double RecallCommitPolicyEvaluator::hypergeometricViolationProbability(
    size_t population,
    size_t weak_records,
    size_t draws,
    size_t allowed_weak_hits) {
    if (weak_records > population) {
        throw std::invalid_argument("weak_records exceeds population");
    }
    draws = std::min(draws, population);
    const size_t min_hits = draws > population - weak_records
                                ? draws - (population - weak_records)
                                : 0;
    const size_t max_hits = std::min(draws, weak_records);
    if (allowed_weak_hits >= max_hits) return 0.0;
    const size_t first = std::max(min_hits, allowed_weak_hits + 1);

    const long double denominator = logChoose(population, draws);
    long double probability = 0.0L;
    for (size_t hits = first; hits <= max_hits; ++hits) {
        const long double log_probability =
            logChoose(weak_records, hits) +
            logChoose(population - weak_records, draws - hits) - denominator;
        probability += std::exp(log_probability);
        if (hits == std::numeric_limits<size_t>::max()) break;
    }
    return static_cast<double>(std::clamp(probability, 0.0L, 1.0L));
}

}  // namespace vdb
