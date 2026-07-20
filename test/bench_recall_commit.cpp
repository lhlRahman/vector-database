#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "../src/core/vector_database.hpp"
#include "../src/utils/atomic_write.hpp"
#include "../src/utils/vecs_io.hpp"

// End-to-end benchmark for the production recall-aware committer. Every trial
// uses the public API and opens a logical crash image through production
// read-only recovery. The partial-survival case uses a controlled process exit
// after WAL-fence sync and before in-memory frontier publication. This is not a
// claim of physical power-loss injection. Exact truth comes from the externally
// recorded visible snapshot, never from recovered state.

namespace {
using Clock = std::chrono::steady_clock;
using Values = std::vector<float>;

struct Options {
    size_t base_records{160};
    size_t writes{25};
    size_t queries{32};
    size_t dimensions{12};
    size_t k{10};
    size_t writers{4};
    size_t repetitions{30};
    size_t ef{32};
    double strict_epsilon{0.20};
    double exchange_epsilon{0.05};
    uint32_t seed{100};
    std::string data_dir;
    std::filesystem::path output_dir{"build/ann_results"};
    bool keep_images{false};
    bool throughput_sweep{false};
    std::vector<size_t> sweep_k_mins{10, 20, 40, 100};
    std::vector<double> sweep_epsilons{0.05, 0.10, 0.20};
    std::vector<size_t> sweep_group_delays_us{750};
    std::vector<size_t> sweep_writers{4};
};

struct DataSet {
    std::string name;
    size_t dimensions{0};
    std::vector<Values> base;
    std::vector<Values> queries;
    std::vector<Values> reserve;
};

enum class Workload { Random, Hot };

enum class CrashFrontier {
    TerminalUnfencedSuffix,
    StrictCapBeforeFence,
    FenceAfterSyncBeforePublish,
};

std::string_view crashFrontierName(CrashFrontier frontier) {
    switch (frontier) {
        case CrashFrontier::TerminalUnfencedSuffix:
            return "terminal-unfenced-suffix";
        case CrashFrontier::StrictCapBeforeFence:
            return "strict-cap-before-fence";
        case CrashFrontier::FenceAfterSyncBeforePublish:
            return "fence-after-sync-before-publish";
    }
    throw std::logic_error("unknown crash frontier");
}

int crashFrontierExitStatus(CrashFrontier frontier) {
    switch (frontier) {
        case CrashFrontier::StrictCapBeforeFence:
            return 87;
        case CrashFrontier::FenceAfterSyncBeforePublish:
            return 86;
        case CrashFrontier::TerminalUnfencedSuffix:
            return -1;
    }
    throw std::logic_error("unknown crash frontier");
}

struct CaseSpec {
    std::string name;
    Workload workload{Workload::Random};
    vdb::RecallPolicy policy{vdb::RecallPolicy::Strict};
    vdb::AckMode ack{vdb::AckMode::Weak};
    double epsilon{0.0};
    size_t record_cap{std::numeric_limits<size_t>::max()};
    std::chrono::milliseconds age_cap{0};
    bool correlation_guard{false};
    CrashFrontier crash_frontier{CrashFrontier::TerminalUnfencedSuffix};
};

struct WriteOp {
    size_t index{0};
    double latency_us{0.0};
    vdb::WriteReceipt receipt;
};

struct QueryOp {
    size_t index{0};
    double latency_us{0.0};
    VectorDatabase::SearchResponse response;
    vdb::DurabilityStatus status;
    double exact_recall{0.0};
};

struct CrashCohortRecord {
    std::string key;
    Values values;
    std::string metadata;
    vdb::WriteReceipt receipt;
    bool expected_to_survive{false};
};

struct RankedKey {
    std::string key;
    double distance{0.0};
};

struct CrashPreQuery {
    size_t query_index{0};
    uint64_t snapshot_lsn{0};
    std::vector<RankedKey> latest;
    std::vector<RankedKey> stable;
};

struct ControlledCrashResult {
    vdb::DurabilityStatus status;
    std::vector<CrashCohortRecord> cohort;
    std::vector<CrashPreQuery> queries;
    int child_exit_status{-1};
};

struct PostRecoverySuffixResult {
    WriteOp operation;
    bool verified{false};
};

struct CrashObservation {
    double membership_risk{0.0};
    double realized_loss{0.0};
    double positive_delta{0.0};
    double amplification{0.0};
    double pre_recall{0.0};
    double post_recall{0.0};
    double answer_churn{0.0};
    double durable_overlap{0.0};
    bool lost_ids_subset_weak{true};
    bool pre_merge_fingerprint_equal{true};
    bool recovery_fingerprint_equal{true};
};

struct TrialResult {
    std::string case_name;
    std::string workload;
    size_t repetition{0};
    uint32_t hnsw_seed{0};
    uint32_t tail_seed{0};
    size_t writer_count{0};
    double write_seconds{0.0};
    double throughput{0.0};
    std::vector<WriteOp> writes;
    std::vector<QueryOp> queries;
    std::vector<double> weak_latencies;
    std::vector<double> stable_latencies;
    std::vector<double> query_latencies;
    double fence_latency_us{0.0};
    double mean_durable{0.0};
    double mean_weak{0.0};
    size_t max_weak{0};
    size_t max_cap{0};
    double max_risk{0.0};
    VectorDatabase::RecallCommitterStatistics committer;
    VectorDatabase::RecallCommitterStatistics timed_committer;
    vdb::RecallCommitPolicyCounters policy;
    std::string crash_frontier{"terminal-unfenced-suffix"};
    uint64_t crash_visible_lsn{0};
    uint64_t crash_durable_lsn{0};
    int crash_child_status{-1};
    std::vector<CrashCohortRecord> crash_cohort;
    std::optional<WriteOp> post_recovery_suffix;
    std::vector<CrashObservation> crash;
    size_t exposed_weak_records{0};
    size_t surviving_weak_records{0};
    size_t lost_weak_records{0};
    size_t stable_losses{0};
    size_t unexpected_weak_survivors{0};
    bool stable_records_unchanged{true};
    bool cohort_records_unchanged{true};
    bool cohort_expectations_ok{true};
    bool post_recovery_suffix_ok{true};
    bool frontier_ok{true};
    bool strict_ok{true};
    bool recovery_ok{true};
    bool has_strict_loss_gap{false};
    bool alarmed{false};
    double alarm_latency_ms{-1.0};
};

struct ThroughputCaseSpec {
    std::string name;
    Workload workload{Workload::Random};
    vdb::AckMode ack{vdb::AckMode::Weak};
    size_t k_min{10};
    double epsilon{0.0};
    double comparison_epsilon{0.0};
    size_t writers{4};
    std::chrono::microseconds group_delay{750};
};

struct ThroughputTrial {
    ThroughputCaseSpec spec;
    size_t repetition{0};
    uint32_t hnsw_seed{0};
    uint32_t tail_seed{0};
    size_t configured_cap{0};
    double write_seconds{0.0};
    double throughput{0.0};
    std::vector<WriteOp> writes;
    std::vector<double> query_latencies;
    size_t latest_queries{0};
    size_t stable_queries{0};
    size_t observed_weak_acks{0};
    size_t observed_stable_acks{0};
    size_t max_weak{0};
    bool all_applied{true};
    VectorDatabase::RecallCommitterStatistics committer;
    vdb::RecallCommitPolicyCounters policy;
};

double elapsedUs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

std::string workloadName(Workload workload) {
    return workload == Workload::Hot ? "hot" : "random";
}

std::string ackName(vdb::AckLevel ack) {
    if (ack == vdb::AckLevel::Weak) return "weak";
    if (ack == vdb::AckLevel::Stable) return "stable";
    return "none";
}

double l2(const Values& left, const Values& right) {
    double sum = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        const double difference = static_cast<double>(left[i]) - right[i];
        sum += difference * difference;
    }
    return sum;
}

std::vector<std::string> responseKeys(const VectorDatabase::SearchResponse& response) {
    std::vector<std::string> keys;
    keys.reserve(response.results.size());
    for (const auto& result : response.results) keys.push_back(result.key);
    return keys;
}

std::vector<RankedKey> responseRanks(const VectorDatabase::SearchResponse& response) {
    std::vector<RankedKey> ranks;
    ranks.reserve(response.results.size());
    for (const auto& result : response.results) {
        ranks.push_back(RankedKey{result.key, result.distance});
    }
    return ranks;
}

std::vector<std::string> rankedKeys(const std::vector<RankedKey>& ranks) {
    std::vector<std::string> keys;
    keys.reserve(ranks.size());
    for (const auto& rank : ranks) keys.push_back(rank.key);
    return keys;
}

std::vector<std::string> mergeStableWithCohort(
    const std::vector<RankedKey>& stable,
    const Values& query,
    const std::vector<CrashCohortRecord>& cohort,
    const std::unordered_set<std::string>& included_cohort,
    size_t k) {
    std::vector<RankedKey> candidates = stable;
    candidates.reserve(stable.size() + included_cohort.size());
    const EuclideanDistance metric;
    for (const auto& record : cohort) {
        if (!included_cohort.contains(record.key)) continue;
        candidates.push_back(RankedKey{
            record.key,
            metric.distance_raw(
                std::span<const float>(query.data(), query.size()),
                std::span<const float>(record.values.data(), record.values.size()))});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        if (left.distance != right.distance) return left.distance < right.distance;
        return left.key < right.key;
    });
    std::vector<std::string> keys;
    std::unordered_set<std::string> seen;
    keys.reserve(std::min(k, candidates.size()));
    for (const auto& candidate : candidates) {
        if (!seen.insert(candidate.key).second) continue;
        keys.push_back(candidate.key);
        if (keys.size() == k) break;
    }
    return keys;
}

bool vectorEquals(const Vector& actual, const Values& expected) {
    if (actual.size() != expected.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual.data_ptr()[i] != expected[i]) return false;
    }
    return true;
}

void writeAll(int fd, std::string_view bytes) {
    while (!bytes.empty()) {
        const ssize_t written = ::write(fd, bytes.data(), bytes.size());
        if (written < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "write crash ledger");
        }
        bytes.remove_prefix(static_cast<size_t>(written));
    }
}

std::string readAll(int fd) {
    std::string bytes;
    char buffer[8192];
    for (;;) {
        const ssize_t count = ::read(fd, buffer, sizeof(buffer));
        if (count < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "read crash ledger");
        }
        if (count == 0) return bytes;
        bytes.append(buffer, static_cast<size_t>(count));
    }
}

double overlap(const std::vector<std::string>& left,
               const std::vector<std::string>& right,
               size_t denominator) {
    if (denominator == 0) return 1.0;
    const std::unordered_set<std::string> expected(right.begin(), right.end());
    size_t count = 0;
    for (const auto& key : left) count += expected.contains(key);
    return static_cast<double>(count) / static_cast<double>(denominator);
}

uint64_t keyFingerprint(const std::vector<std::string>& keys) {
    uint64_t hash = 1469598103934665603ULL;
    for (const auto& key : keys) {
        for (const unsigned char value : key) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        hash ^= 0xff;
        hash *= 1099511628211ULL;
    }
    return hash;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double location = p * static_cast<double>(values.size() - 1);
    const size_t low = static_cast<size_t>(location);
    const size_t high = std::min(low + 1, values.size() - 1);
    const double fraction = location - static_cast<double>(low);
    return values[low] + fraction * (values[high] - values[low]);
}

void copyTree(const std::filesystem::path& source,
              const std::filesystem::path& destination) {
    std::filesystem::remove_all(destination);
    std::filesystem::create_directories(destination);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(source)) {
        const auto relative = std::filesystem::relative(entry.path(), source);
        const auto target = destination / relative;
        if (entry.is_directory()) {
            std::filesystem::create_directories(target);
        } else if (entry.is_regular_file()) {
            std::filesystem::create_directories(target.parent_path());
            std::filesystem::copy_file(entry.path(), target,
                                       std::filesystem::copy_options::overwrite_existing);
        }
    }
}

uint32_t persistedSeed(const std::filesystem::path& root) {
    std::ifstream input(root / "manifest.txt");
    std::string line;
    while (std::getline(input, line)) {
        constexpr std::string_view prefix = "hnsw_seed=";
        if (line.starts_with(prefix)) {
            return static_cast<uint32_t>(std::stoul(line.substr(prefix.size())));
        }
    }
    throw std::runtime_error("manifest has no hnsw_seed: " + root.string());
}

DataSet makeSynthetic(const Options& options) {
    DataSet data;
    data.name = "clustered-synthetic";
    data.dimensions = options.dimensions;
    std::mt19937 generator(71237);
    std::normal_distribution<float> center_distribution(0.0f, 2.0f);
    std::normal_distribution<float> base_noise(0.0f, 0.30f);
    std::normal_distribution<float> query_noise(0.0f, 0.08f);
    const size_t cluster_count = 20;
    std::vector<Values> centers(cluster_count, Values(data.dimensions));
    for (auto& center : centers) {
        for (float& value : center) value = center_distribution(generator);
    }
    auto make_row = [&](size_t cluster, auto& noise) {
        Values row = centers[cluster % cluster_count];
        for (float& value : row) value += noise(generator);
        return row;
    };
    for (size_t i = 0; i < options.base_records; ++i) {
        data.base.push_back(make_row(i, base_noise));
    }
    for (size_t i = 0; i < options.queries; ++i) {
        data.queries.push_back(make_row(i, query_noise));
    }
    for (size_t i = 0; i < options.writes * options.repetitions + 64; ++i) {
        data.reserve.push_back(make_row(i * 7 + 3, base_noise));
    }
    return data;
}

DataSet loadData(const Options& options) {
    if (options.data_dir.empty()) return makeSynthetic(options);
    std::string base_path;
    std::string query_path;
    for (const auto& entry : std::filesystem::directory_iterator(options.data_dir)) {
        const std::string name = entry.path().filename().string();
        if (name.ends_with("_base.fvecs")) base_path = entry.path().string();
        if (name.ends_with("_query.fvecs")) query_path = entry.path().string();
    }
    const auto base = vecs_io::load_fvecs(base_path);
    const auto queries = vecs_io::load_fvecs(query_path);
    if (base.d != queries.d || base.n < options.base_records + options.writes ||
        queries.n < options.queries) {
        throw std::runtime_error("dataset is too small or has mismatched dimensions");
    }
    DataSet data;
    data.name = std::filesystem::path(options.data_dir).filename().string();
    data.dimensions = base.d;
    for (size_t i = 0; i < options.base_records; ++i) {
        data.base.emplace_back(base.row(i), base.row(i) + base.d);
    }
    for (size_t i = 0; i < options.queries; ++i) {
        data.queries.emplace_back(queries.row(i), queries.row(i) + queries.d);
    }
    const size_t reserve_count = std::min(base.n - options.base_records,
                                          options.writes * options.repetitions + 64);
    for (size_t i = 0; i < reserve_count; ++i) {
        const float* row = base.row(options.base_records + i);
        data.reserve.emplace_back(row, row + base.d);
    }
    return data;
}

std::vector<Values> makeTail(const DataSet& data,
                             const Options& options,
                             Workload workload,
                             size_t repetition) {
    std::vector<Values> tail;
    tail.reserve(options.writes);
    if (workload == Workload::Random) {
        std::mt19937 generator(9001 + static_cast<uint32_t>(repetition));
        std::vector<size_t> order(data.reserve.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), generator);
        for (size_t i = 0; i < options.writes; ++i) {
            tail.push_back(data.reserve[order[i % order.size()]]);
        }
        return tail;
    }

    std::normal_distribution<float> noise(0.0f, 0.0005f);
    std::mt19937 generator(45001 + static_cast<uint32_t>(repetition));
    const size_t hot_queries = std::min<size_t>(4, data.queries.size());
    for (size_t i = 0; i < options.writes; ++i) {
        Values row = data.queries[i % hot_queries];
        for (float& value : row) value += noise(generator);
        tail.push_back(std::move(row));
    }
    return tail;
}

vdb::RecallCommitConfig makeConfig(const CaseSpec& spec,
                                   const Options& options,
                                   uint32_t seed) {
    vdb::RecallCommitConfig config;
    config.enabled = true;
    config.policy = spec.policy;
    config.epsilon = spec.epsilon;
    config.k_min = options.k;
    config.max_tail_records = spec.record_cap;
    config.max_tail_bytes = 64u * 1024u * 1024u;
    config.max_tail_age = spec.age_cap;
    config.group_delay = std::chrono::microseconds(750);
    config.hnsw_seed = seed;
    config.correlation_guard_enabled = spec.correlation_guard;
    config.correlation_min_queries = 4;
    config.correlation_min_expected_hits = 0.0;
    config.correlation_enrichment_threshold = 1.2;
    config.correlation_cusum_drift = 0.0;
    config.correlation_cusum_threshold = 1.5;
    return config;
}

std::vector<std::string> exactTopK(const Values& query,
                                   const DataSet& data,
                                   const std::vector<Values>& tail,
                                   const std::vector<WriteOp>& operations,
                                   uint64_t snapshot_lsn,
                                   size_t k,
                                   const std::vector<CrashCohortRecord>* crash_cohort = nullptr) {
    std::vector<std::pair<double, std::string>> candidates;
    candidates.reserve(data.base.size() + tail.size() +
                       (crash_cohort == nullptr ? 0 : crash_cohort->size()));
    std::unordered_set<std::string> candidate_keys;
    auto add_candidate = [&](double distance, std::string key) {
        if (candidate_keys.insert(key).second) {
            candidates.emplace_back(distance, std::move(key));
        }
    };
    for (size_t i = 0; i < data.base.size(); ++i) {
        add_candidate(l2(query, data.base[i]), "base-" + std::to_string(i));
    }
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& receipt = operations[i].receipt;
        if (receipt.applied && receipt.lsn <= snapshot_lsn) {
            add_candidate(l2(query, tail[i]), "tail-" + std::to_string(i));
        }
    }
    if (crash_cohort != nullptr) {
        for (const auto& record : *crash_cohort) {
            if (record.receipt.applied && record.receipt.lsn <= snapshot_lsn) {
                add_candidate(l2(query, record.values), record.key);
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first < right.first;
        return left.second < right.second;
    });
    std::vector<std::string> result;
    for (size_t i = 0; i < std::min(k, candidates.size()); ++i) {
        result.push_back(candidates[i].second);
    }
    return result;
}

std::filesystem::path buildBaseImage(const DataSet& data,
                                     const Options& options,
                                     const std::filesystem::path& root,
                                     uint32_t seed,
                                     bool reverse_order,
                                     const std::string& image_name) {
    const auto path = root / image_name;
    std::filesystem::remove_all(path);
    CaseSpec stable{"base-build", Workload::Random, vdb::RecallPolicy::Strict,
                    vdb::AckMode::Stable, 0.0};
    VectorDatabase database(data.dimensions, VectorDatabase::SearchMode::HNSW,
                            false, false, {}, false, 0, path.string());
    database.configureHNSW(16, 100, options.ef, seed);
    database.configureSegmentedStorage(data.base.size() + options.writes + 128);
    database.configureRecallCommit(makeConfig(stable, options, seed));
    database.initialize();

    std::atomic<size_t> next{0};
    std::atomic<bool> go{false};
    std::atomic<size_t> failed{0};
    std::vector<std::thread> writers;
    for (size_t writer = 0; writer < std::min(options.writers, data.base.size()); ++writer) {
        writers.emplace_back([&] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (;;) {
                const size_t position = next.fetch_add(1);
                if (position >= data.base.size()) break;
                const size_t i = reverse_order ? data.base.size() - 1 - position : position;
                const auto receipt = database.insertWithAck(
                    Vector(data.base[i]), "base-" + std::to_string(i), "base",
                    vdb::AckMode::Stable);
                if (!receipt.applied || receipt.actual_ack != vdb::AckLevel::Stable) ++failed;
            }
        });
    }
    go.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    if (failed != 0) throw std::runtime_error("failed to build durable base image");
    (void)database.durabilityFence();
    database.sealMutableSegment();
    database.shutdown();
    return path;
}

bool validateChangedSeedControl(const DataSet& data,
                                const Options& options,
                                const std::filesystem::path& root,
                                const std::filesystem::path& production_base,
                                uint32_t production_seed) {
    const uint32_t changed_seed = production_seed + 1009;
    const auto changed = buildBaseImage(
        data, options, root, changed_seed, true, "changed-seed-base");
    VectorDatabase production(data.dimensions, VectorDatabase::SearchMode::HNSW,
                              false, false, {}, false, 0, production_base.string(),
                              VectorDatabase::StorageEngine::Segmented,
                              vdb::OpenMode::ReadOnlyRecovery);
    VectorDatabase control(data.dimensions, VectorDatabase::SearchMode::HNSW,
                           false, false, {}, false, 0, changed.string(),
                           VectorDatabase::StorageEngine::Segmented,
                           vdb::OpenMode::ReadOnlyRecovery);
    production.configureHNSW(16, 100, options.ef, production_seed);
    control.configureHNSW(16, 100, options.ef, changed_seed);
    production.initialize();
    control.initialize();
    bool answer_drift = false;
    for (const auto& values : data.queries) {
        const auto left = responseKeys(production.similaritySearch(
            Vector(values), options.k, vdb::ReadVisibility::Stable));
        const auto right = responseKeys(control.similaritySearch(
            Vector(values), options.k, vdb::ReadVisibility::Stable));
        answer_drift = answer_drift || keyFingerprint(left) != keyFingerprint(right);
    }
    const bool topology_fingerprint_equal = persistedSeed(changed) == production_seed;
    const bool accepted = topology_fingerprint_equal && !answer_drift;
    production.shutdown();
    control.shutdown();
    std::cout << "negative_control changed_seed=" << changed_seed
              << " answer_drift=" << (answer_drift ? 1 : 0)
              << " topology_fingerprint_mismatch="
              << (!topology_fingerprint_equal ? 1 : 0)
              << " validator_rejected=" << (!accepted ? 1 : 0) << '\n';
    return !accepted;
}

ControlledCrashResult runControlledStrictCrash(
    const DataSet& data,
    const Options& options,
    const CaseSpec& spec,
    const std::filesystem::path& database_path,
    uint32_t graph_seed,
    const vdb::RecallCommitConfig& config) {
    const size_t cohort_size = static_cast<size_t>(
        std::floor(spec.epsilon * static_cast<double>(options.k)));
    const double cap_loss = static_cast<double>(cohort_size) /
                            static_cast<double>(options.k);
    if (spec.policy != vdb::RecallPolicy::Strict || cohort_size == 0 ||
        spec.crash_frontier == CrashFrontier::TerminalUnfencedSuffix) {
        throw std::logic_error("controlled crash requires a binding strict frontier");
    }
    const std::string frontier_name(crashFrontierName(spec.crash_frontier));
    const size_t crash_queries = std::min<size_t>(12, data.queries.size());
    int ledger_pipe[2];
    if (::pipe(ledger_pipe) != 0) {
        throw std::system_error(errno, std::generic_category(), "pipe crash ledger");
    }

    const pid_t child = ::fork();
    if (child < 0) {
        ::close(ledger_pipe[0]);
        ::close(ledger_pipe[1]);
        throw std::system_error(errno, std::generic_category(), "fork crash child");
    }
    if (child == 0) {
        ::close(ledger_pipe[0]);
        try {
            ::unsetenv("VDB_COMMITTER_FAILPOINT");
            auto* database = new VectorDatabase(
                data.dimensions, VectorDatabase::SearchMode::HNSW,
                false, false, {}, false, 0, database_path.string());
            database->configureHNSW(16, 100, options.ef, graph_seed);
            database->configureSegmentedStorage(data.base.size() + options.writes + 128);
            database->configureRecallCommit(config);
            database->initialize();

            std::vector<vdb::WriteReceipt> receipts;
            receipts.reserve(cohort_size);
            for (size_t i = 0; i < cohort_size; ++i) {
                receipts.push_back(database->insertWithAck(
                    Vector(data.queries.front()),
                    "crash-weak-" + std::to_string(i),
                    frontier_name, vdb::AckMode::Weak));
            }
            const auto status = database->durabilityStatus();
            for (const auto& receipt : receipts) {
                if (!receipt.applied || receipt.actual_ack != vdb::AckLevel::Weak ||
                    receipt.lsn <= status.durable_lsn ||
                    receipt.lsn > status.visible_lsn) {
                    throw std::runtime_error(
                        "crash cohort was not entirely weak at the observed frontier");
                }
            }
            if (status.weak_records != cohort_size ||
                status.policy_record_cap != cohort_size ||
                std::abs(status.estimated_recall_loss - cap_loss) > 1e-12 ||
                status.visible_lsn <= status.durable_lsn) {
                throw std::runtime_error(
                    "observed U_pre does not exactly bind the strict crash cap");
            }

            std::ostringstream ledger;
            ledger << std::setprecision(17);
            ledger << "STATUS " << status.appended_lsn << ' ' << status.visible_lsn
                   << ' ' << status.durable_lsn << ' ' << status.visible_records
                   << ' ' << status.durable_records << ' ' << status.weak_records
                   << ' ' << status.weak_bytes << ' ' << status.policy_record_cap
                   << ' ' << status.estimated_recall_loss << '\n';
            for (size_t i = 0; i < receipts.size(); ++i) {
                const auto& receipt = receipts[i];
                ledger << "RECEIPT " << i << " crash-weak-" << i << ' '
                       << receipt.lsn << ' ' << receipt.visible_lsn << ' '
                       << receipt.durable_lsn << ' ' << receipt.durable_count << ' '
                       << receipt.weak_count << ' ' << receipt.policy_cap << ' '
                       << receipt.risk_estimate << ' ' << (receipt.provisional ? 1 : 0)
                       << '\n';
            }
            for (size_t q = 0; q < crash_queries; ++q) {
                const size_t query_index = spec.workload == Workload::Hot
                                               ? q % std::min<size_t>(4, data.queries.size())
                                               : q;
                const Vector query(data.queries[query_index]);
                const auto latest = database->similaritySearch(
                    query, options.k, vdb::ReadVisibility::Latest);
                const auto stable = database->similaritySearch(
                    query, options.k, vdb::ReadVisibility::Stable);
                if (latest.snapshot_lsn != status.visible_lsn ||
                    stable.snapshot_lsn != status.durable_lsn) {
                    throw std::runtime_error("crash query did not use the recorded frontier");
                }
                ledger << "QUERY " << q << ' ' << query_index << ' '
                       << latest.snapshot_lsn << ' ' << latest.results.size();
                for (const auto& result : latest.results) {
                    ledger << ' ' << result.key << ' ' << result.distance;
                }
                ledger << ' ' << stable.results.size();
                for (const auto& result : stable.results) {
                    ledger << ' ' << result.key << ' ' << result.distance;
                }
                ledger << '\n';
            }
            writeAll(ledger_pipe[1], ledger.str());
            if (spec.crash_frontier == CrashFrontier::StrictCapBeforeFence) {
                _exit(crashFrontierExitStatus(spec.crash_frontier));
            }
            if (::setenv("VDB_COMMITTER_FAILPOINT", "fence-after-sync", 1) != 0) {
                throw std::system_error(errno, std::generic_category(),
                                        "set crash failpoint");
            }
            (void)database->durabilityFence();
            writeAll(ledger_pipe[1], "ERROR fence-after-sync failpoint returned\n");
            _exit(3);
        } catch (const std::exception& error) {
            try {
                std::string message = error.what();
                std::replace(message.begin(), message.end(), '\n', ' ');
                writeAll(ledger_pipe[1], "ERROR " + message + "\n");
            } catch (...) {
            }
            _exit(2);
        } catch (...) {
            _exit(2);
        }
    }

    ::close(ledger_pipe[1]);
    const std::string ledger = readAll(ledger_pipe[0]);
    ::close(ledger_pipe[0]);
    int wait_status = 0;
    while (::waitpid(child, &wait_status, 0) < 0) {
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "wait crash child");
        }
    }
    const int child_status = WIFEXITED(wait_status)
                                 ? WEXITSTATUS(wait_status)
                                 : (WIFSIGNALED(wait_status)
                                        ? 128 + WTERMSIG(wait_status)
                                        : 255);
    const int expected_child_status = crashFrontierExitStatus(spec.crash_frontier);
    if (child_status != expected_child_status) {
        throw std::runtime_error(
            frontier_name + " child status=" + std::to_string(child_status) +
            " ledger=" + ledger);
    }

    ControlledCrashResult result;
    result.child_exit_status = child_status;
    result.queries.resize(crash_queries);
    std::vector<bool> saw_query(crash_queries, false);
    std::vector<bool> saw_receipt(cohort_size, false);
    bool saw_status = false;
    std::istringstream input(ledger);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string kind;
        fields >> kind;
        if (kind == "ERROR") {
            throw std::runtime_error("crash child reported: " + line.substr(kind.size() + 1));
        }
        if (kind == "STATUS") {
            fields >> result.status.appended_lsn >> result.status.visible_lsn
                   >> result.status.durable_lsn >> result.status.visible_records
                   >> result.status.durable_records >> result.status.weak_records
                   >> result.status.weak_bytes >> result.status.policy_record_cap
                   >> result.status.estimated_recall_loss;
            saw_status = true;
        } else if (kind == "RECEIPT") {
            size_t ordinal = 0;
            std::string key;
            int provisional = 0;
            vdb::WriteReceipt receipt;
            fields >> ordinal >> key >> receipt.lsn >> receipt.visible_lsn
                   >> receipt.durable_lsn >> receipt.durable_count
                   >> receipt.weak_count >> receipt.policy_cap
                   >> receipt.risk_estimate >> provisional;
            if (ordinal >= cohort_size || saw_receipt[ordinal]) {
                throw std::runtime_error("invalid duplicate crash receipt");
            }
            receipt.applied = true;
            receipt.requested_ack = vdb::AckMode::Weak;
            receipt.actual_ack = vdb::AckLevel::Weak;
            receipt.provisional = provisional != 0;
            result.cohort.push_back(CrashCohortRecord{
                std::move(key), data.queries.front(), frontier_name, receipt,
                spec.crash_frontier == CrashFrontier::FenceAfterSyncBeforePublish});
            saw_receipt[ordinal] = true;
        } else if (kind == "QUERY") {
            size_t ordinal = 0;
            size_t latest_count = 0;
            fields >> ordinal;
            if (ordinal >= crash_queries || saw_query[ordinal]) {
                throw std::runtime_error("invalid duplicate crash query");
            }
            auto& query = result.queries[ordinal];
            fields >> query.query_index >> query.snapshot_lsn >> latest_count;
            query.latest.resize(latest_count);
            for (auto& rank : query.latest) fields >> rank.key >> rank.distance;
            size_t stable_count = 0;
            fields >> stable_count;
            query.stable.resize(stable_count);
            for (auto& rank : query.stable) fields >> rank.key >> rank.distance;
            saw_query[ordinal] = true;
        } else if (!kind.empty()) {
            throw std::runtime_error("unknown crash ledger row: " + kind);
        }
        if (!fields) throw std::runtime_error("malformed crash ledger row: " + line);
    }
    if (!saw_status ||
        !std::all_of(saw_receipt.begin(), saw_receipt.end(), [](bool saw) { return saw; }) ||
        !std::all_of(saw_query.begin(), saw_query.end(), [](bool saw) { return saw; })) {
        throw std::runtime_error("incomplete crash child ledger");
    }
    std::sort(result.cohort.begin(), result.cohort.end(), [](const auto& left, const auto& right) {
        return left.key < right.key;
    });
    return result;
}

PostRecoverySuffixResult runPostRecoverySuffix(
    const DataSet& data,
    const Options& options,
    const std::filesystem::path& resume_path,
    uint32_t graph_seed,
    const vdb::RecallCommitConfig& config,
    const std::vector<CrashCohortRecord>& crash_cohort) {
    constexpr std::string_view key = "post-recovery-suffix";
    constexpr std::string_view metadata = "post-recovery-suffix";
    const Values& values = data.queries.back();

    VectorDatabase resumed(data.dimensions, VectorDatabase::SearchMode::HNSW,
                           false, false, {}, false, 0, resume_path.string());
    resumed.configureHNSW(16, 100, options.ef, graph_seed);
    resumed.configureSegmentedStorage(data.base.size() + options.writes + 128);
    resumed.configureRecallCommit(config);
    resumed.initialize();
    const auto begin = Clock::now();
    auto receipt = resumed.insertWithAck(
        Vector(values), std::string(key), std::string(metadata), vdb::AckMode::Stable);
    const auto end = Clock::now();
    const auto immediate = resumed.inspectRecord(
        std::string(key), vdb::ReadVisibility::Stable);
    const bool stable_ack = receipt.applied &&
                            receipt.actual_ack == vdb::AckLevel::Stable &&
                            receipt.durable_lsn >= receipt.lsn && immediate &&
                            vectorEquals(immediate->vector, values) &&
                            immediate->metadata == metadata &&
                            immediate->lsn == receipt.lsn && !immediate->provisional;
    resumed.shutdown();

    VectorDatabase verified(data.dimensions, VectorDatabase::SearchMode::HNSW,
                            false, false, {}, false, 0, resume_path.string(),
                            VectorDatabase::StorageEngine::Segmented,
                            vdb::OpenMode::ReadOnlyRecovery);
    verified.configureHNSW(16, 100, options.ef, graph_seed);
    verified.configureRecallCommit(config);
    verified.initialize();
    const auto reopened = verified.inspectRecord(
        std::string(key), vdb::ReadVisibility::Stable);
    const bool persisted = reopened && vectorEquals(reopened->vector, values) &&
                           reopened->metadata == metadata &&
                           reopened->lsn == receipt.lsn && !reopened->provisional;
    bool survivors_intact = true;
    for (const auto& survivor : crash_cohort) {
        const auto record = verified.inspectRecord(
            survivor.key, vdb::ReadVisibility::Stable);
        survivors_intact = survivors_intact && record &&
                           vectorEquals(record->vector, survivor.values) &&
                           record->metadata == survivor.metadata &&
                           record->lsn == survivor.receipt.lsn &&
                           !record->provisional;
    }
    verified.shutdown();
    return PostRecoverySuffixResult{
        WriteOp{0, elapsedUs(begin, end), receipt},
        stable_ack && persisted && survivors_intact};
}

TrialResult runTrial(const DataSet& data,
                     const Options& options,
                     const CaseSpec& spec,
                     size_t repetition,
                     const std::filesystem::path& base_image,
                     uint32_t graph_seed,
                     const std::filesystem::path& root) {
    TrialResult trial;
    trial.case_name = spec.name;
    trial.workload = workloadName(spec.workload);
    trial.repetition = repetition;
    trial.hnsw_seed = graph_seed;
    trial.tail_seed = static_cast<uint32_t>(
        (spec.workload == Workload::Hot ? 45001 : 9001) + repetition);
    trial.writer_count = options.writers;
    const auto database_path = root / (spec.name + "-" + std::to_string(repetition));
    const auto crash_path = root / (spec.name + "-" + std::to_string(repetition) + "-crash");
    const auto resume_path = root / (spec.name + "-" + std::to_string(repetition) + "-resume");
    copyTree(base_image, database_path);
    const auto config = makeConfig(spec, options, graph_seed);
    const auto tail = makeTail(data, options, spec.workload, repetition);

    VectorDatabase database(data.dimensions, VectorDatabase::SearchMode::HNSW,
                            false, false, {}, false, 0, database_path.string());
    database.configureHNSW(16, 100, options.ef, graph_seed);
    database.configureSegmentedStorage(data.base.size() + options.writes + 128);
    database.configureRecallCommit(config);
    database.initialize();

    trial.writes.resize(options.writes);
    std::atomic<size_t> next{0};
    std::atomic<bool> go{false};
    std::atomic<int64_t> alarm_us{-1};
    std::mutex error_mutex;
    std::exception_ptr error;
    auto capture_error = [&] {
        std::lock_guard lock(error_mutex);
        if (!error) error = std::current_exception();
    };

    std::vector<std::thread> writers;
    for (size_t writer = 0; writer < options.writers; ++writer) {
        writers.emplace_back([&, writer] {
            try {
                while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= options.writes) break;
                    const auto begin = Clock::now();
                    auto receipt = database.insertWithAck(
                        Vector(tail[i]), "tail-" + std::to_string(i), spec.name,
                        spec.ack);
                    const auto end = Clock::now();
                    trial.writes[i] = WriteOp{i, elapsedUs(begin, end), receipt};
                    std::this_thread::sleep_for(std::chrono::microseconds(100 + writer * 15));
                }
            } catch (...) {
                capture_error();
            }
        });
    }

    const auto workload_start = Clock::now();
    std::thread query_thread([&] {
        try {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (size_t i = 0; i < options.queries; ++i) {
                const size_t query_index = spec.workload == Workload::Hot
                                               ? i % std::min<size_t>(4, data.queries.size())
                                               : i % data.queries.size();
                const auto begin = Clock::now();
                auto response = database.similaritySearch(
                    Vector(data.queries[query_index]), options.k,
                    vdb::ReadVisibility::Latest);
                const auto end = Clock::now();
                auto status = database.durabilityStatus();
                trial.queries.push_back(QueryOp{
                    query_index, elapsedUs(begin, end), std::move(response), status, 0.0});
                const auto correlation = database.recallPolicyStatistics().correlation;
                if (correlation.alarmed && alarm_us.load() < 0) {
                    alarm_us.store(static_cast<int64_t>(elapsedUs(workload_start, end)));
                }
                std::this_thread::sleep_for(std::chrono::microseconds(120));
            }
        } catch (...) {
            capture_error();
        }
    });

    go.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    const auto writers_end = Clock::now();
    trial.timed_committer = database.recallCommitterStatistics();
    query_thread.join();
    if (error) std::rethrow_exception(error);
    trial.write_seconds = std::chrono::duration<double>(writers_end - workload_start).count();
    trial.throughput = static_cast<double>(options.writes) / trial.write_seconds;

    const size_t crash_queries = std::min<size_t>(12, data.queries.size());
    std::vector<CrashPreQuery> pre_queries;
    vdb::DurabilityStatus pre_status;
    bool database_shutdown = false;
    const bool controlled_frontier =
        spec.crash_frontier != CrashFrontier::TerminalUnfencedSuffix;
    if (controlled_frontier) {
        if (spec.policy != vdb::RecallPolicy::Strict ||
            std::floor(spec.epsilon * static_cast<double>(options.k)) < 2.0) {
            throw std::runtime_error(
                "controlled frontier requires a strict cap of at least two records");
        }

        // Leave a fully fenced timed prefix, then make the crash child the only
        // process with this image open. Its cap-sized query-targeted cohort is
        // entirely in U_pre. One frontier exits before fencing; the other exits
        // after fence sync but before publishing the in-memory durable frontier.
        const auto fence_begin = Clock::now();
        (void)database.durabilityFence();
        trial.fence_latency_us = elapsedUs(fence_begin, Clock::now());
        trial.committer = database.recallCommitterStatistics();
        trial.policy = database.recallPolicyStatistics();
        trial.alarmed = trial.policy.correlation.alarmed;
        database.shutdown();
        database_shutdown = true;

        trial.crash_frontier = crashFrontierName(spec.crash_frontier);
        auto controlled = runControlledStrictCrash(
            data, options, spec, database_path, graph_seed, config);
        pre_status = controlled.status;
        pre_queries = std::move(controlled.queries);
        trial.crash_cohort = std::move(controlled.cohort);
        trial.crash_child_status = controlled.child_exit_status;
        copyTree(database_path, crash_path);
    } else {
        pre_status = database.durabilityStatus();
        const bool asynchronous_fence_pending =
            pre_status.correlation_alarm || spec.age_cap.count() > 0;
        if (asynchronous_fence_pending && pre_status.weak_records != 0) {
            if (!database.waitUntilDurable(
                    pre_status.visible_lsn, std::chrono::seconds(2))) {
                throw std::runtime_error(
                    "asynchronous policy fence missed the crash-frontier deadline");
            }
            pre_status = database.durabilityStatus();
        }
        for (const auto& operation : trial.writes) {
            if (operation.receipt.applied &&
                operation.receipt.lsn > pre_status.durable_lsn) {
                trial.crash_cohort.push_back(CrashCohortRecord{
                    "tail-" + std::to_string(operation.index),
                    tail[operation.index], spec.name, operation.receipt, false});
            }
        }
        copyTree(database_path, crash_path);
        pre_queries.reserve(crash_queries);
        for (size_t q = 0; q < crash_queries; ++q) {
            const size_t query_index = spec.workload == Workload::Hot
                                           ? q % std::min<size_t>(4, data.queries.size())
                                           : q;
            const Vector query(data.queries[query_index]);
            const auto latest = database.similaritySearch(
                query, options.k, vdb::ReadVisibility::Latest);
            const auto stable = database.similaritySearch(
                query, options.k, vdb::ReadVisibility::Stable);
            pre_queries.push_back(CrashPreQuery{
                query_index, latest.snapshot_lsn,
                responseRanks(latest), responseRanks(stable)});
        }
    }

    trial.crash_visible_lsn = pre_status.visible_lsn;
    trial.crash_durable_lsn = pre_status.durable_lsn;

    VectorDatabase recovered(data.dimensions, VectorDatabase::SearchMode::HNSW,
                             false, false, {}, false, 0, crash_path.string(),
                             VectorDatabase::StorageEngine::Segmented,
                             vdb::OpenMode::ReadOnlyRecovery);
    recovered.configureHNSW(16, 100, options.ef, graph_seed);
    recovered.configureRecallCommit(config);
    recovered.initialize();

    std::unordered_set<std::string> exposed_weak;
    std::unordered_set<std::string> surviving_weak;
    std::unordered_set<std::string> lost_weak;
    trial.exposed_weak_records = trial.crash_cohort.size();
    for (const auto& record : trial.crash_cohort) exposed_weak.insert(record.key);
    for (size_t i = 0; i < data.base.size(); ++i) {
        const auto record = recovered.inspectRecord(
            "base-" + std::to_string(i), vdb::ReadVisibility::Stable);
        if (!record) {
            ++trial.stable_losses;
            trial.stable_records_unchanged = false;
        } else if (!vectorEquals(record->vector, data.base[i]) ||
                   record->metadata != "base" || record->provisional) {
            trial.stable_records_unchanged = false;
        }
    }
    for (const auto& operation : trial.writes) {
        if (!operation.receipt.applied) continue;
        const std::string key = "tail-" + std::to_string(operation.index);
        if (operation.receipt.lsn <= pre_status.durable_lsn) {
            const auto record = recovered.inspectRecord(key, vdb::ReadVisibility::Stable);
            if (!record) {
                ++trial.stable_losses;
                trial.stable_records_unchanged = false;
            } else if (!vectorEquals(record->vector, tail[operation.index]) ||
                       record->metadata != spec.name ||
                       record->lsn != operation.receipt.lsn || record->provisional) {
                trial.stable_records_unchanged = false;
            }
        }
    }
    for (const auto& record : trial.crash_cohort) {
        const auto latest_record = recovered.inspectRecord(
            record.key, vdb::ReadVisibility::Latest);
        const bool present = latest_record.has_value();
        if (present) {
            ++trial.surviving_weak_records;
            surviving_weak.insert(record.key);
            const auto stable_record = recovered.inspectRecord(
                record.key, vdb::ReadVisibility::Stable);
            const auto& inspected = record.expected_to_survive
                                        ? stable_record
                                        : latest_record;
            if (!inspected || !vectorEquals(inspected->vector, record.values) ||
                inspected->metadata != record.metadata ||
                inspected->lsn != record.receipt.lsn || inspected->provisional ||
                (record.expected_to_survive && !stable_record)) {
                trial.cohort_records_unchanged = false;
            }
        } else {
            ++trial.lost_weak_records;
            lost_weak.insert(record.key);
        }
        if (record.expected_to_survive != present) trial.cohort_expectations_ok = false;
        if (!record.expected_to_survive && present) ++trial.unexpected_weak_survivors;
    }

    for (size_t q = 0; q < crash_queries; ++q) {
        const auto& pre = pre_queries.at(q);
        const size_t query_index = pre.query_index;
        const Vector query(data.queries[query_index]);
        const auto post = recovered.similaritySearch(query, options.k,
                                                      vdb::ReadVisibility::Latest);
        const auto truth = exactTopK(data.queries[query_index], data, tail,
                                     trial.writes, pre.snapshot_lsn, options.k,
                                     &trial.crash_cohort);
        const auto pre_keys = rankedKeys(pre.latest);
        const auto post_keys = responseKeys(post);
        const auto stable_keys = rankedKeys(pre.stable);
        const auto expected_pre = mergeStableWithCohort(
            pre.stable, data.queries[query_index], trial.crash_cohort,
            exposed_weak, options.k);
        const auto expected_recovery = mergeStableWithCohort(
            pre.stable, data.queries[query_index], trial.crash_cohort,
            surviving_weak, options.k);
        size_t exposed_weak_truth = 0;
        size_t lost_weak_truth = 0;
        for (const auto& key : truth) {
            exposed_weak_truth += exposed_weak.contains(key);
            lost_weak_truth += lost_weak.contains(key);
        }
        const double denominator = static_cast<double>(std::max<size_t>(1, truth.size()));
        const double pre_recall = overlap(pre_keys, truth, truth.size());
        const double post_recall = overlap(post_keys, truth, truth.size());
        bool subset = true;
        const std::unordered_set<std::string> post_set(post_keys.begin(), post_keys.end());
        for (const auto& key : pre_keys) {
            if (!post_set.contains(key) && !lost_weak.contains(key)) subset = false;
        }
        CrashObservation observation;
        observation.membership_risk =
            static_cast<double>(exposed_weak_truth) / denominator;
        observation.realized_loss = static_cast<double>(lost_weak_truth) / denominator;
        observation.positive_delta = std::max(0.0, pre_recall - post_recall);
        observation.amplification = std::max(
            0.0, observation.positive_delta - observation.realized_loss);
        observation.pre_recall = pre_recall;
        observation.post_recall = post_recall;
        observation.answer_churn = 1.0 - overlap(pre_keys, post_keys, options.k);
        observation.durable_overlap = overlap(stable_keys, post_keys, options.k);
        observation.lost_ids_subset_weak = subset;
        observation.pre_merge_fingerprint_equal =
            keyFingerprint(expected_pre) == keyFingerprint(pre_keys);
        observation.recovery_fingerprint_equal =
            keyFingerprint(expected_recovery) == keyFingerprint(post_keys);
        trial.has_strict_loss_gap = trial.has_strict_loss_gap ||
                                    observation.realized_loss + 1e-12 <
                                        observation.membership_risk;
        trial.crash.push_back(observation);
    }

    for (auto& query : trial.queries) {
        const auto truth = exactTopK(data.queries[query.index], data, tail, trial.writes,
                                     query.response.snapshot_lsn, options.k);
        query.exact_recall = overlap(responseKeys(query.response), truth, truth.size());
        trial.query_latencies.push_back(query.latency_us);
    }
    for (const auto& operation : trial.writes) {
        if (operation.receipt.actual_ack == vdb::AckLevel::Weak) {
            trial.weak_latencies.push_back(operation.latency_us);
        } else if (operation.receipt.actual_ack == vdb::AckLevel::Stable) {
            trial.stable_latencies.push_back(operation.latency_us);
        }
        trial.mean_durable += operation.receipt.durable_count;
        trial.mean_weak += operation.receipt.weak_count;
        trial.max_weak = std::max(trial.max_weak, operation.receipt.weak_count);
        trial.max_cap = std::max(trial.max_cap, operation.receipt.policy_cap);
        trial.max_risk = std::max(trial.max_risk, operation.receipt.risk_estimate);
        trial.frontier_ok = trial.frontier_ok &&
            operation.receipt.visible_lsn >= operation.receipt.durable_lsn;
    }
    trial.mean_durable /= static_cast<double>(std::max<size_t>(1, trial.writes.size()));
    trial.mean_weak /= static_cast<double>(std::max<size_t>(1, trial.writes.size()));
    if (alarm_us.load() >= 0) trial.alarm_latency_ms = alarm_us.load() / 1000.0;

    if (!database_shutdown) {
        trial.policy = database.recallPolicyStatistics();
        trial.alarmed = trial.policy.correlation.alarmed;
        const auto fence_begin = Clock::now();
        (void)database.durabilityFence();
        trial.fence_latency_us = elapsedUs(fence_begin, Clock::now());
        trial.committer = database.recallCommitterStatistics();
    }
    trial.frontier_ok = trial.frontier_ok && pre_status.appended_lsn >= pre_status.visible_lsn &&
                        pre_status.visible_lsn >= pre_status.durable_lsn &&
                        pre_status.weak_records == trial.exposed_weak_records;
    for (const auto& record : trial.crash_cohort) {
        trial.frontier_ok = trial.frontier_ok && record.receipt.applied &&
                            record.receipt.actual_ack == vdb::AckLevel::Weak &&
                            record.receipt.lsn > pre_status.durable_lsn &&
                            record.receipt.lsn <= pre_status.visible_lsn;
    }
    trial.recovery_ok = trial.stable_losses == 0 &&
                        trial.unexpected_weak_survivors == 0 &&
                        trial.stable_records_unchanged &&
                        trial.cohort_records_unchanged &&
                        trial.cohort_expectations_ok;
    for (const auto& observation : trial.crash) {
        trial.recovery_ok = trial.recovery_ok && observation.lost_ids_subset_weak &&
                            observation.pre_merge_fingerprint_equal &&
                            observation.recovery_fingerprint_equal &&
                            observation.amplification <= 1e-12;
        if (spec.policy == vdb::RecallPolicy::Strict) {
            trial.strict_ok = trial.strict_ok &&
                observation.positive_delta <= observation.realized_loss + 1e-12 &&
                observation.realized_loss <= observation.membership_risk + 1e-12 &&
                observation.membership_risk <= spec.epsilon + 1e-12;
        }
    }
    if (spec.crash_frontier == CrashFrontier::FenceAfterSyncBeforePublish) {
        trial.recovery_ok = trial.recovery_ok && trial.exposed_weak_records == 2 &&
                            trial.surviving_weak_records >= 1 &&
                            trial.crash_child_status == 86 && trial.has_strict_loss_gap;
    } else if (spec.crash_frontier == CrashFrontier::StrictCapBeforeFence) {
        const size_t strict_cap = static_cast<size_t>(
            std::floor(spec.epsilon * static_cast<double>(options.k)));
        const double cap_loss = static_cast<double>(strict_cap) /
                                static_cast<double>(options.k);
        const bool reached_cap_loss = std::any_of(
            trial.crash.begin(), trial.crash.end(), [&](const CrashObservation& observation) {
                return std::abs(observation.membership_risk - cap_loss) <= 1e-12 &&
                       std::abs(observation.realized_loss - cap_loss) <= 1e-12 &&
                       std::abs(observation.positive_delta - cap_loss) <= 1e-12;
            });
        trial.recovery_ok = trial.recovery_ok &&
                            trial.exposed_weak_records == strict_cap &&
                            trial.surviving_weak_records == 0 &&
                            trial.lost_weak_records == strict_cap &&
                            trial.crash_child_status == 87 && reached_cap_loss;
    }
    trial.strict_ok = trial.strict_ok && trial.policy.cap_overshoots == 0;
    recovered.shutdown();
    if (spec.crash_frontier == CrashFrontier::FenceAfterSyncBeforePublish) {
        copyTree(crash_path, resume_path);
        auto suffix = runPostRecoverySuffix(
            data, options, resume_path, graph_seed, config, trial.crash_cohort);
        trial.post_recovery_suffix = suffix.operation;
        trial.post_recovery_suffix_ok = suffix.verified;
        trial.recovery_ok = trial.recovery_ok && trial.post_recovery_suffix_ok;
    }
    if (!database_shutdown) database.shutdown();
    if (!options.keep_images) {
        std::filesystem::remove_all(database_path);
        std::filesystem::remove_all(crash_path);
        std::filesystem::remove_all(resume_path);
    }
    return trial;
}

ThroughputTrial runThroughputTrial(const DataSet& data,
                                   const Options& options,
                                   const ThroughputCaseSpec& spec,
                                   size_t repetition,
                                   const std::filesystem::path& base_image,
                                   uint32_t graph_seed,
                                   const std::filesystem::path& root) {
    ThroughputTrial trial;
    trial.spec = spec;
    trial.repetition = repetition;
    trial.hnsw_seed = graph_seed;
    trial.tail_seed = static_cast<uint32_t>(
        (spec.workload == Workload::Hot ? 45001 : 9001) + repetition);

    const auto database_path = root / (spec.name + "-" + std::to_string(repetition));
    copyTree(base_image, database_path);
    CaseSpec config_spec{spec.name, spec.workload, vdb::RecallPolicy::Strict,
                         spec.ack, spec.epsilon};
    auto config = makeConfig(config_spec, options, graph_seed);
    config.k_min = spec.k_min;
    config.group_delay = spec.group_delay;
    trial.configured_cap =
        vdb::RecallCommitPolicyEvaluator::policyRecordCap(config, data.base.size());
    const auto tail = makeTail(data, options, spec.workload, repetition);

    VectorDatabase database(data.dimensions, VectorDatabase::SearchMode::HNSW,
                            false, false, {}, false, 0, database_path.string());
    database.configureHNSW(16, 100, options.ef, graph_seed);
    database.configureSegmentedStorage(data.base.size() + options.writes + 128);
    database.configureRecallCommit(config);
    database.initialize();

    trial.writes.resize(options.writes);
    std::atomic<size_t> next{0};
    std::atomic<bool> go{false};
    std::mutex error_mutex;
    std::exception_ptr error;
    auto capture_error = [&] {
        std::lock_guard lock(error_mutex);
        if (!error) error = std::current_exception();
    };

    std::vector<std::thread> writers;
    for (size_t writer = 0; writer < spec.writers; ++writer) {
        writers.emplace_back([&, writer] {
            try {
                while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= options.writes) break;
                    const auto begin = Clock::now();
                    auto receipt = database.insertWithAck(
                        Vector(tail[i]), "tail-" + std::to_string(i), "throughput-sweep",
                        spec.ack);
                    const auto end = Clock::now();
                    trial.writes[i] = WriteOp{i, elapsedUs(begin, end), receipt};
                    std::this_thread::sleep_for(std::chrono::microseconds(100 + writer * 15));
                }
            } catch (...) {
                capture_error();
            }
        });
    }

    std::thread query_thread([&] {
        try {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (size_t i = 0; i < options.queries; ++i) {
                const size_t query_index = spec.workload == Workload::Hot
                                               ? i % std::min<size_t>(4, data.queries.size())
                                               : i % data.queries.size();
                const auto begin = Clock::now();
                const auto response = database.similaritySearch(
                    Vector(data.queries[query_index]), spec.k_min,
                    vdb::ReadVisibility::Latest);
                const auto end = Clock::now();
                trial.query_latencies.push_back(elapsedUs(begin, end));
                if (response.effective_visibility == vdb::ReadVisibility::Latest) {
                    ++trial.latest_queries;
                } else {
                    ++trial.stable_queries;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(120));
            }
        } catch (...) {
            capture_error();
        }
    });

    const auto workload_start = Clock::now();
    go.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    const auto writers_end = Clock::now();
    trial.committer = database.recallCommitterStatistics();
    trial.policy = database.recallPolicyStatistics();
    query_thread.join();
    if (error) std::rethrow_exception(error);
    trial.write_seconds = std::chrono::duration<double>(writers_end - workload_start).count();
    trial.throughput = static_cast<double>(options.writes) / trial.write_seconds;

    for (const auto& operation : trial.writes) {
        trial.all_applied = trial.all_applied && operation.receipt.applied;
        trial.max_weak = std::max(trial.max_weak, operation.receipt.weak_count);
        if (operation.receipt.actual_ack == vdb::AckLevel::Weak) {
            ++trial.observed_weak_acks;
        } else if (operation.receipt.actual_ack == vdb::AckLevel::Stable) {
            ++trial.observed_stable_acks;
        }
    }

    (void)database.durabilityFence();
    database.shutdown();
    if (!options.keep_images) std::filesystem::remove_all(database_path);
    return trial;
}

std::string throughputCaseName(vdb::AckMode ack,
                               Workload workload,
                               size_t k_min,
                               double epsilon,
                               size_t writers,
                               size_t group_delay_us) {
    std::ostringstream name;
    name << (ack == vdb::AckMode::Stable ? "stable" : "strict") << '-'
         << workloadName(workload) << "-k" << k_min;
    if (ack == vdb::AckMode::Weak) {
        name << "-e" << std::llround(epsilon * 1000000.0);
    } else {
        name << "-for-e" << std::llround(epsilon * 1000000.0);
    }
    name << "-g" << group_delay_us << "-w" << writers;
    return name.str();
}

void writeThroughputSweepCsv(const std::filesystem::path& path,
                             const std::vector<ThroughputTrial>& trials) {
    std::ofstream output(path);
    output << std::setprecision(17);
    output << "case,workload,repetition,hnsw_seed,tail_seed,requested_ack,writers,writes,"
              "queries,k_min,epsilon,comparison_epsilon,configured_cap,group_delay_us,"
              "write_seconds,writes_per_s,"
              "weak_acks,stable_acks,max_weak,binding,weak_p50_us,weak_p95_us,weak_p99_us,"
              "stable_p50_us,stable_p95_us,stable_p99_us,query_p50_us,query_p95_us,query_p99_us,"
              "latest_queries,stable_queries,timed_sync_attempts,timed_sync_successes,"
              "timed_sync_failures,timed_records_synced,policy_fences,follower_requests,"
              "fence_then_retry,strict_rejections,cap_overshoots,all_applied\n";
    for (const auto& trial : trials) {
        std::vector<double> weak_latencies;
        std::vector<double> stable_latencies;
        for (const auto& operation : trial.writes) {
            if (operation.receipt.actual_ack == vdb::AckLevel::Weak) {
                weak_latencies.push_back(operation.latency_us);
            } else if (operation.receipt.actual_ack == vdb::AckLevel::Stable) {
                stable_latencies.push_back(operation.latency_us);
            }
        }
        const bool binding = trial.configured_cap >= 1 && trial.observed_weak_acks > 0;
        output << trial.spec.name << ',' << workloadName(trial.spec.workload) << ','
               << trial.repetition << ',' << trial.hnsw_seed << ',' << trial.tail_seed << ','
               << (trial.spec.ack == vdb::AckMode::Stable ? "stable" : "weak") << ','
               << trial.spec.writers << ',' << trial.writes.size() << ','
               << trial.query_latencies.size() << ',' << trial.spec.k_min << ','
               << trial.spec.epsilon << ',' << trial.spec.comparison_epsilon << ','
               << trial.configured_cap << ','
               << trial.spec.group_delay.count() << ',' << trial.write_seconds << ','
               << trial.throughput << ',' << trial.observed_weak_acks << ','
               << trial.observed_stable_acks << ',' << trial.max_weak << ','
               << (binding ? 1 : 0) << ',' << percentile(weak_latencies, .50) << ','
               << percentile(weak_latencies, .95) << ',' << percentile(weak_latencies, .99)
               << ',' << percentile(stable_latencies, .50) << ','
               << percentile(stable_latencies, .95) << ','
               << percentile(stable_latencies, .99) << ','
               << percentile(trial.query_latencies, .50) << ','
               << percentile(trial.query_latencies, .95) << ','
               << percentile(trial.query_latencies, .99) << ',' << trial.latest_queries << ','
               << trial.stable_queries << ',' << trial.committer.sync_attempts << ','
               << trial.committer.sync_successes << ',' << trial.committer.sync_failures << ','
               << trial.committer.records_synced << ',' << trial.committer.policy_fences << ','
               << trial.committer.follower_requests << ',' << trial.policy.fence_then_retry
               << ',' << trial.policy.strict_rejections << ',' << trial.policy.cap_overshoots
               << ',' << (trial.all_applied ? 1 : 0) << '\n';
    }
}

void writeThroughputSweepOperationsCsv(const std::filesystem::path& path,
                                       const std::vector<ThroughputTrial>& trials) {
    std::ofstream output(path);
    output << std::setprecision(17);
    output << "case,workload,repetition,hnsw_seed,tail_seed,requested_ack,writers,writes,"
              "queries,k_min,epsilon,comparison_epsilon,configured_cap,group_delay_us,index,"
              "actual_ack,latency_us,"
              "lsn,visible_lsn,durable_lsn,durable_records,weak_records,receipt_cap,risk\n";
    for (const auto& trial : trials) {
        for (const auto& operation : trial.writes) {
            const auto& receipt = operation.receipt;
            output << trial.spec.name << ',' << workloadName(trial.spec.workload) << ','
                   << trial.repetition << ',' << trial.hnsw_seed << ',' << trial.tail_seed << ','
                   << (trial.spec.ack == vdb::AckMode::Stable ? "stable" : "weak") << ','
                   << trial.spec.writers << ',' << trial.writes.size() << ','
                   << trial.query_latencies.size() << ',' << trial.spec.k_min << ','
                   << trial.spec.epsilon << ',' << trial.spec.comparison_epsilon << ','
                   << trial.configured_cap << ','
                   << trial.spec.group_delay.count() << ',' << operation.index << ','
                   << ackName(receipt.actual_ack) << ',' << operation.latency_us << ','
                   << receipt.lsn << ',' << receipt.visible_lsn << ',' << receipt.durable_lsn
                   << ',' << receipt.durable_count << ',' << receipt.weak_count << ','
                   << receipt.policy_cap << ',' << receipt.risk_estimate << '\n';
        }
    }
}

bool runThroughputSweep(const DataSet& data,
                        const Options& options,
                        const std::vector<std::filesystem::path>& bases,
                        const std::vector<uint32_t>& graph_seeds,
                        const std::filesystem::path& root) {
    std::vector<ThroughputCaseSpec> cases;
    for (const size_t writers : options.sweep_writers) {
        for (const size_t delay_us : options.sweep_group_delays_us) {
            for (const size_t k_min : options.sweep_k_mins) {
                for (const Workload workload : {Workload::Random, Workload::Hot}) {
                    for (const double epsilon : options.sweep_epsilons) {
                        cases.push_back(ThroughputCaseSpec{
                            throughputCaseName(vdb::AckMode::Stable, workload, k_min, epsilon,
                                               writers, delay_us),
                            workload, vdb::AckMode::Stable, k_min, 0.0, epsilon, writers,
                            std::chrono::microseconds(delay_us)});
                        cases.push_back(ThroughputCaseSpec{
                            throughputCaseName(vdb::AckMode::Weak, workload, k_min, epsilon,
                                               writers, delay_us),
                            workload, vdb::AckMode::Weak, k_min, epsilon, epsilon, writers,
                            std::chrono::microseconds(delay_us)});
                    }
                }
            }
        }
    }

    std::vector<ThroughputTrial> trials;
    trials.reserve(options.repetitions * cases.size());
    bool all_ok = true;
    for (size_t repetition = 0; repetition < options.repetitions; ++repetition) {
        std::vector<size_t> pair_order(cases.size() / 2);
        std::iota(pair_order.begin(), pair_order.end(), 0);
        std::mt19937 order_rng(options.seed + static_cast<uint32_t>(repetition));
        std::shuffle(pair_order.begin(), pair_order.end(), order_rng);
        const size_t graph_index = repetition % bases.size();
        for (const size_t pair_index : pair_order) {
            const bool strict_first = (order_rng() & 1u) != 0;
            for (size_t position = 0; position < 2; ++position) {
                const size_t pair_offset = strict_first ? 1 - position : position;
                const auto& spec = cases[pair_index * 2 + pair_offset];
                auto trial = runThroughputTrial(data, options, spec, repetition,
                                                bases[graph_index], graph_seeds[graph_index], root);
                const bool binding =
                    trial.configured_cap >= 1 && trial.observed_weak_acks > 0;
                const bool cap_ok =
                    trial.configured_cap == 0 || trial.policy.cap_overshoots == 0;
                bool trial_ok = trial.all_applied && cap_ok &&
                                trial.latest_queries == options.queries &&
                                trial.max_weak <= trial.configured_cap;
                if (spec.ack == vdb::AckMode::Stable) {
                    trial_ok = trial_ok && trial.observed_weak_acks == 0 &&
                               trial.observed_stable_acks == options.writes;
                } else if (trial.configured_cap == 0) {
                    trial_ok = trial_ok && trial.observed_weak_acks == 0 &&
                               trial.observed_stable_acks == options.writes;
                } else {
                    trial_ok = trial_ok && binding;
                }
                all_ok = all_ok && trial_ok;
                std::cout << std::left << std::setw(38) << spec.name
                          << " rep=" << repetition << " writes/s=" << std::fixed
                          << std::setprecision(1) << trial.throughput
                          << " cap=" << trial.configured_cap
                          << " weak=" << trial.observed_weak_acks
                          << " stable=" << trial.observed_stable_acks
                          << " maxW=" << trial.max_weak
                          << " syncs=" << trial.committer.sync_successes
                          << " binding=" << (binding ? 1 : 0)
                          << " ok=" << (trial_ok ? 1 : 0) << '\n';
                trials.push_back(std::move(trial));
            }
        }
    }

    const auto aggregate_path = options.output_dir / "recall_committer_throughput_sweep.csv";
    const auto operations_path =
        options.output_dir / "recall_committer_throughput_sweep_operations.csv";
    writeThroughputSweepCsv(aggregate_path, trials);
    writeThroughputSweepOperationsCsv(operations_path, trials);

    std::cout << "\npaired fixed-graph timing-repetition min/median/max\n";
    for (const auto& spec : cases) {
        std::vector<double> throughputs;
        for (const auto& trial : trials) {
            if (trial.spec.name == spec.name) throughputs.push_back(trial.throughput);
        }
        std::cout << std::left << std::setw(34) << spec.name << " writes/s=["
                  << *std::min_element(throughputs.begin(), throughputs.end()) << ','
                  << percentile(throughputs, .50) << ','
                  << *std::max_element(throughputs.begin(), throughputs.end()) << "]\n";
    }
    std::cout << "aggregate_csv=" << aggregate_path << " operations_csv=" << operations_path
              << " sweep_invariants_ok=" << (all_ok ? 1 : 0) << '\n';
    return all_ok;
}

void writeRawCsv(const std::filesystem::path& path,
                 const std::vector<TrialResult>& trials) {
    std::ofstream output(path);
    output << "case,workload,repetition,hnsw_seed,tail_seed,operation,index,ack,latency_us,lsn,"
              "visible_lsn,durable_lsn,durable_records,weak_records,cap,risk,"
              "snapshot_lsn,exact_recall,tail_evaluations,crash_frontier,expected_recovery,"
              "crash_child_status\n";
    for (const auto& trial : trials) {
        for (const auto& operation : trial.writes) {
            const auto& receipt = operation.receipt;
            output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                   << ',' << trial.hnsw_seed << ',' << trial.tail_seed
                   << ",write," << operation.index << ','
                   << ackName(receipt.actual_ack) << ',' << operation.latency_us << ','
                   << receipt.lsn << ',' << receipt.visible_lsn << ','
                   << receipt.durable_lsn << ',' << receipt.durable_count << ','
                   << receipt.weak_count << ',' << receipt.policy_cap << ','
                   << receipt.risk_estimate << ",,,," << trial.crash_frontier << ",,"
                   << trial.crash_child_status << '\n';
        }
        for (size_t i = 0; i < trial.queries.size(); ++i) {
            const auto& query = trial.queries[i];
            output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                   << ',' << trial.hnsw_seed << ',' << trial.tail_seed
                   << ",query," << i << ",," << query.latency_us
                   << ",," << query.response.snapshot_lsn << ','
                   << query.response.durable_lsn << ',' << query.status.durable_records
                   << ',' << query.status.weak_records << ',' << query.status.policy_record_cap
                   << ',' << query.status.estimated_recall_loss << ','
                   << query.response.snapshot_lsn << ',' << query.exact_recall << ','
                   << query.response.exact_tail_distance_evaluations << ','
                   << trial.crash_frontier << ",," << trial.crash_child_status << '\n';
        }
        if (trial.crash_frontier != "terminal-unfenced-suffix") {
            output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                   << ',' << trial.hnsw_seed << ',' << trial.tail_seed
                   << ",timed-prefix-fence,0,stable," << trial.fence_latency_us
                   << ",0,0,0,0,0,0,0,0,0,0," << trial.crash_frontier << ",,"
                   << trial.crash_child_status << '\n';
        }
        if (trial.crash_frontier != "terminal-unfenced-suffix") {
            for (size_t i = 0; i < trial.crash_cohort.size(); ++i) {
                const auto& record = trial.crash_cohort[i];
                const auto& receipt = record.receipt;
                output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                       << ',' << trial.hnsw_seed << ',' << trial.tail_seed
                       << ",crash-cohort," << i << ',' << ackName(receipt.actual_ack)
                       << ",0," << receipt.lsn << ',' << receipt.visible_lsn << ','
                       << receipt.durable_lsn << ',' << receipt.durable_count << ','
                       << receipt.weak_count << ',' << receipt.policy_cap << ','
                       << receipt.risk_estimate << ",,,," << trial.crash_frontier << ','
                       << (record.expected_to_survive ? "survive" : "lost") << ','
                       << trial.crash_child_status << '\n';
            }
            if (trial.crash_frontier == "fence-after-sync-before-publish") {
                output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                       << ',' << trial.hnsw_seed << ',' << trial.tail_seed
                       << ",crash-fence,0,none,0," << trial.crash_visible_lsn << ','
                       << trial.crash_visible_lsn << ',' << trial.crash_durable_lsn
                       << ",0," << trial.exposed_weak_records
                       << ",0,0,,,," << trial.crash_frontier
                       << ",child-exit-after-sync," << trial.crash_child_status << '\n';
                if (!trial.post_recovery_suffix) {
                    throw std::runtime_error(
                        "post-sync crash workflow has no post-recovery suffix");
                }
                const auto& suffix = *trial.post_recovery_suffix;
                const auto& receipt = suffix.receipt;
                output << trial.case_name << ',' << trial.workload << ','
                       << trial.repetition << ',' << trial.hnsw_seed << ','
                       << trial.tail_seed << ",post-recovery-suffix,0,"
                       << ackName(receipt.actual_ack) << ',' << suffix.latency_us << ','
                       << receipt.lsn << ',' << receipt.visible_lsn << ','
                       << receipt.durable_lsn << ',' << receipt.durable_count << ','
                       << receipt.weak_count << ',' << receipt.policy_cap << ','
                       << receipt.risk_estimate << ",,,," << trial.crash_frontier
                       << ",workflow-resumed," << trial.crash_child_status << '\n';
            }
        }
        if (trial.crash_frontier == "terminal-unfenced-suffix") {
            output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                   << ',' << trial.hnsw_seed << ',' << trial.tail_seed
                   << ",cleanup-fence,0,stable," << trial.fence_latency_us
                   << ",0,0,0,0,0,0,0,0,0,0," << trial.crash_frontier << ",,"
                   << trial.crash_child_status << '\n';
        }
    }
}

void writeAggregateCsv(const std::filesystem::path& path,
                       const std::vector<TrialResult>& trials) {
    std::ofstream output(path);
    output << "case,workload,repetition,hnsw_seed,tail_seed,writers,writes_per_s,"
              "weak_p50_us,weak_p95_us,"
              "weak_p99_us,stable_p50_us,stable_p95_us,stable_p99_us,fence_us,"
              "query_p50_us,query_p95_us,query_p99_us,mean_D,mean_W,max_W,max_cap,"
              "max_risk,timed_sync_successes,total_sync_attempts,total_sync_successes,"
              "total_sync_failures,timed_syncs_per_s,total_records_per_sync,"
              "policy_fences,age_fences,overshoots,enrichment,alarm,alarm_latency_ms,"
              "crash_frontier,crash_visible_lsn,crash_durable_lsn,crash_child_status,"
              "exposed_weak_records,surviving_weak_records,lost_weak_records,"
              "has_L_lt_M,M_max,L_max,delta_max,amplification_max,stable_losses,weak_survivors,"
              "stable_records_unchanged,cohort_records_unchanged,cohort_expectations_ok,"
              "post_recovery_suffix_ok,frontier_ok,recovery_ok,strict_ok\n";
    for (const auto& trial : trials) {
        double m = 0.0, l = 0.0, delta = 0.0, amplification = 0.0;
        for (const auto& observation : trial.crash) {
            m = std::max(m, observation.membership_risk);
            l = std::max(l, observation.realized_loss);
            delta = std::max(delta, observation.positive_delta);
            amplification = std::max(amplification, observation.amplification);
        }
        const double records_per_sync = trial.committer.sync_successes == 0
                                            ? 0.0
                                            : static_cast<double>(trial.committer.records_synced) /
                                                  trial.committer.sync_successes;
        output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
               << ',' << trial.hnsw_seed << ',' << trial.tail_seed
               << ',' << trial.writer_count << ',' << trial.throughput << ','
               << percentile(trial.weak_latencies, .50) << ','
               << percentile(trial.weak_latencies, .95) << ','
               << percentile(trial.weak_latencies, .99) << ','
               << percentile(trial.stable_latencies, .50) << ','
               << percentile(trial.stable_latencies, .95) << ','
               << percentile(trial.stable_latencies, .99) << ','
               << trial.fence_latency_us << ',' << percentile(trial.query_latencies, .50)
               << ',' << percentile(trial.query_latencies, .95) << ','
               << percentile(trial.query_latencies, .99) << ',' << trial.mean_durable
               << ',' << trial.mean_weak << ',' << trial.max_weak << ',' << trial.max_cap
               << ',' << trial.max_risk << ',' << trial.timed_committer.sync_successes
               << ',' << trial.committer.sync_attempts << ','
               << trial.committer.sync_successes << ',' << trial.committer.sync_failures
               << ',' << trial.timed_committer.sync_successes / trial.write_seconds
               << ',' << records_per_sync << ',' << trial.committer.policy_fences << ','
               << trial.committer.age_fences << ',' << trial.policy.cap_overshoots << ','
               << trial.policy.correlation.enrichment << ',' << (trial.alarmed ? 1 : 0)
               << ',' << trial.alarm_latency_ms << ',' << trial.crash_frontier << ','
               << trial.crash_visible_lsn << ',' << trial.crash_durable_lsn << ','
               << trial.crash_child_status << ','
               << trial.exposed_weak_records << ',' << trial.surviving_weak_records << ','
               << trial.lost_weak_records << ',' << (trial.has_strict_loss_gap ? 1 : 0)
               << ',' << m << ',' << l << ',' << delta
               << ',' << amplification << ',' << trial.stable_losses << ','
               << trial.unexpected_weak_survivors << ','
               << (trial.stable_records_unchanged ? 1 : 0) << ','
               << (trial.cohort_records_unchanged ? 1 : 0) << ','
               << (trial.cohort_expectations_ok ? 1 : 0) << ','
               << (trial.post_recovery_suffix_ok ? 1 : 0) << ','
               << (trial.frontier_ok ? 1 : 0)
               << ',' << (trial.recovery_ok ? 1 : 0) << ',' << (trial.strict_ok ? 1 : 0)
               << '\n';
    }
}

void writeCrashCsv(const std::filesystem::path& path,
                   const std::vector<TrialResult>& trials) {
    std::ofstream output(path);
    output << "case,workload,repetition,hnsw_seed,tail_seed,crash_frontier,"
              "crash_visible_lsn,crash_durable_lsn,crash_child_status,"
              "exposed_weak_records,surviving_weak_records,lost_weak_records,query,"
              "M,L,delta_positive,amplification,"
              "pre_recall,post_recall,answer_churn,pre_stable_overlap,"
              "lost_ids_subset_weak,pre_merge_fingerprint_equal,"
              "recovery_fingerprint_equal,stable_records_unchanged,"
              "cohort_records_unchanged,cohort_expectations_ok,"
              "post_recovery_suffix_ok\n";
    for (const auto& trial : trials) {
        for (size_t i = 0; i < trial.crash.size(); ++i) {
            const auto& observation = trial.crash[i];
            output << trial.case_name << ',' << trial.workload << ','
                   << trial.repetition << ',' << trial.hnsw_seed << ',' << trial.tail_seed
                   << ',' << trial.crash_frontier << ',' << trial.crash_visible_lsn << ','
                   << trial.crash_durable_lsn << ',' << trial.crash_child_status << ','
                   << trial.exposed_weak_records << ','
                   << trial.surviving_weak_records << ',' << trial.lost_weak_records << ','
                   << i << ','
                   << observation.membership_risk << ',' << observation.realized_loss
                   << ',' << observation.positive_delta << ','
                   << observation.amplification << ',' << observation.pre_recall << ','
                   << observation.post_recall << ',' << observation.answer_churn << ','
                   << observation.durable_overlap << ','
                   << (observation.lost_ids_subset_weak ? 1 : 0) << ','
                   << (observation.pre_merge_fingerprint_equal ? 1 : 0) << ','
                   << (observation.recovery_fingerprint_equal ? 1 : 0) << ','
                   << (trial.stable_records_unchanged ? 1 : 0) << ','
                   << (trial.cohort_records_unchanged ? 1 : 0) << ','
                   << (trial.cohort_expectations_ok ? 1 : 0) << ','
                   << (trial.post_recovery_suffix_ok ? 1 : 0) << '\n';
        }
    }
}

std::vector<size_t> parseSizeList(const std::string& value, const std::string& option) {
    std::vector<size_t> result;
    std::stringstream input(value);
    std::string token;
    while (std::getline(input, token, ',')) {
        if (token.empty()) throw std::invalid_argument("empty value in " + option);
        size_t consumed = 0;
        const auto parsed = std::stoull(token, &consumed);
        if (consumed != token.size()) throw std::invalid_argument("invalid value in " + option);
        result.push_back(static_cast<size_t>(parsed));
    }
    if (result.empty()) throw std::invalid_argument(option + " must not be empty");
    return result;
}

std::vector<double> parseDoubleList(const std::string& value, const std::string& option) {
    std::vector<double> result;
    std::stringstream input(value);
    std::string token;
    while (std::getline(input, token, ',')) {
        if (token.empty()) throw std::invalid_argument("empty value in " + option);
        size_t consumed = 0;
        const double parsed = std::stod(token, &consumed);
        if (consumed != token.size()) throw std::invalid_argument("invalid value in " + option);
        result.push_back(parsed);
    }
    if (result.empty()) throw std::invalid_argument(option + " must not be empty");
    return result;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + argument);
            return argv[i];
        };
        if (argument == "--data") options.data_dir = next();
        else if (argument == "--n" || argument == "--base") options.base_records = std::stoul(next());
        else if (argument == "--writes") options.writes = std::stoul(next());
        else if (argument == "--q" || argument == "--queries") options.queries = std::stoul(next());
        else if (argument == "--dim") options.dimensions = std::stoul(next());
        else if (argument == "--k") options.k = std::stoul(next());
        else if (argument == "--writers") options.writers = std::stoul(next());
        else if (argument == "--repetitions") options.repetitions = std::stoul(next());
        else if (argument == "--ef") options.ef = std::stoul(next());
        else if (argument == "--epsilon") options.strict_epsilon = std::stod(next());
        else if (argument == "--exchange-epsilon") options.exchange_epsilon = std::stod(next());
        else if (argument == "--output") options.output_dir = next();
        else if (argument == "--keep-images") options.keep_images = true;
        else if (argument == "--throughput-sweep") options.throughput_sweep = true;
        else if (argument == "--sweep-k-mins") {
            options.sweep_k_mins = parseSizeList(next(), argument);
        } else if (argument == "--sweep-epsilons") {
            options.sweep_epsilons = parseDoubleList(next(), argument);
        } else if (argument == "--sweep-group-delays-us") {
            options.sweep_group_delays_us = parseSizeList(next(), argument);
        } else if (argument == "--sweep-writers") {
            options.sweep_writers = parseSizeList(next(), argument);
        }
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.writers == 0 || options.writes == 0 || options.queries == 0 ||
        options.base_records == 0 || options.k == 0 || options.repetitions == 0) {
        throw std::invalid_argument("benchmark counts must be nonzero");
    }
    if (options.base_records < options.k) {
        throw std::invalid_argument("base records must be at least k");
    }
    if (!options.throughput_sweep &&
        std::floor(options.strict_epsilon * static_cast<double>(options.k)) < 2.0) {
        throw std::invalid_argument(
            "strict epsilon*k must admit two records for partial-survival validation");
    }
    if (options.throughput_sweep) {
        for (const size_t k_min : options.sweep_k_mins) {
            if (k_min == 0 || k_min > options.base_records) {
                throw std::invalid_argument("sweep k_min must be in [1, base records]");
            }
        }
        for (const double epsilon : options.sweep_epsilons) {
            if (!std::isfinite(epsilon) || epsilon < 0.0 || epsilon >= 1.0) {
                throw std::invalid_argument("sweep epsilon must be finite and in [0, 1)");
            }
        }
        for (const size_t writers : options.sweep_writers) {
            if (writers == 0) throw std::invalid_argument("sweep writers must be nonzero");
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        vdb::io::set_full_fsync(true);
        const Options options = parseOptions(argc, argv);
        const DataSet data = loadData(options);
        const auto timestamp = Clock::now().time_since_epoch().count();
        const auto root = std::filesystem::temp_directory_path() /
                          ("vdb-recall-bench-" + std::to_string(timestamp));
        std::filesystem::create_directories(root);
        std::filesystem::create_directories(options.output_dir);
        std::cout << "recall-committer benchmark dataset=" << data.name
                  << " base=" << data.base.size() << " writes=" << options.writes
                  << " queries=" << options.queries << " dim=" << data.dimensions
                  << " k=" << options.k << " writers=" << options.writers
                  << " repetitions=" << options.repetitions << '\n';

        const std::vector<uint32_t> graph_seeds{options.seed, options.seed + 17};
        std::vector<std::filesystem::path> bases;
        bases.reserve(graph_seeds.size());
        for (size_t i = 0; i < graph_seeds.size(); ++i) {
            bases.push_back(buildBaseImage(
                data, options, root, graph_seeds[i], false,
                "base-seed-" + std::to_string(graph_seeds[i])));
        }
        if (options.throughput_sweep) {
            const bool sweep_ok = runThroughputSweep(data, options, bases, graph_seeds, root);
            if (!options.keep_images) std::filesystem::remove_all(root);
            return sweep_ok ? 0 : 2;
        }
        const bool negative_control_tripped =
            validateChangedSeedControl(data, options, root, bases.front(), graph_seeds.front());
        if (!negative_control_tripped) {
            throw std::runtime_error("changed-seed topology control was not rejected");
        }

        const std::vector<CaseSpec> cases{
            {"stable-group", Workload::Random, vdb::RecallPolicy::Strict,
             vdb::AckMode::Stable, 0.0},
            {"fixed-count", Workload::Random, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, 0.90, 4},
            {"fixed-time", Workload::Random, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, 0.90, std::numeric_limits<size_t>::max(),
             std::chrono::milliseconds(2)},
            {"strict-random", Workload::Random, vdb::RecallPolicy::Strict,
             vdb::AckMode::Weak, options.strict_epsilon,
             std::numeric_limits<size_t>::max(), std::chrono::milliseconds(0), false,
             CrashFrontier::StrictCapBeforeFence},
            {"strict-hot", Workload::Hot, vdb::RecallPolicy::Strict,
             vdb::AckMode::Weak, options.strict_epsilon,
             std::numeric_limits<size_t>::max(), std::chrono::milliseconds(0), false,
             CrashFrontier::FenceAfterSyncBeforePublish},
            {"exchange-random", Workload::Random, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, options.exchange_epsilon},
            {"exchange-hot", Workload::Hot, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, options.exchange_epsilon},
            {"exchange-hot-guard", Workload::Hot, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, options.exchange_epsilon,
             std::numeric_limits<size_t>::max(), std::chrono::milliseconds(0), true},
        };

        std::vector<TrialResult> trials;
        std::set<uint32_t> observed_graph_seeds;
        bool observed_strict_loss_gap = false;
        bool observed_partial_survival = false;
        bool terminal_loss_control_tripped = false;
        size_t post_recovery_suffixes = 0;
        size_t strict_cap_loss_images = 0;
        for (size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            std::vector<size_t> case_order(cases.size());
            std::iota(case_order.begin(), case_order.end(), 0);
            std::mt19937 case_order_rng(
                options.seed + static_cast<uint32_t>(repetition));
            std::shuffle(case_order.begin(), case_order.end(), case_order_rng);
            for (const size_t case_index : case_order) {
                const auto& spec = cases[case_index];
                const size_t graph_index = (repetition + case_index) % bases.size();
                auto trial = runTrial(data, options, spec, repetition,
                                      bases[graph_index], graph_seeds[graph_index], root);
                double max_m = 0.0, max_l = 0.0, max_delta = 0.0, max_amp = 0.0;
                for (const auto& observation : trial.crash) {
                    max_m = std::max(max_m, observation.membership_risk);
                    max_l = std::max(max_l, observation.realized_loss);
                    max_delta = std::max(max_delta, observation.positive_delta);
                    max_amp = std::max(max_amp, observation.amplification);
                }
                std::cout << std::left << std::setw(20) << spec.name
                          << " rep=" << repetition << " writes/s=" << std::fixed
                          << std::setprecision(1) << trial.throughput
                          << " weak_p95_us=" << percentile(trial.weak_latencies, .95)
                          << " query_p95_us=" << percentile(trial.query_latencies, .95)
                          << " maxW=" << trial.max_weak << " M=" << max_m
                          << " L=" << max_l << " Delta+=" << max_delta
                          << " amp=" << max_amp << " timed_syncs="
                          << trial.timed_committer.sync_successes << " total_syncs="
                          << trial.committer.sync_successes << " alarm="
                          << (trial.alarmed ? 1 : 0) << " strict_ok="
                          << (trial.strict_ok ? 1 : 0) << " recovery_ok="
                          << (trial.recovery_ok ? 1 : 0) << " suffix_ok="
                          << (trial.post_recovery_suffix_ok ? 1 : 0) << '\n';
                observed_graph_seeds.insert(trial.hnsw_seed);
                observed_strict_loss_gap = observed_strict_loss_gap ||
                                           trial.has_strict_loss_gap;
                observed_partial_survival = observed_partial_survival ||
                    (spec.crash_frontier == CrashFrontier::FenceAfterSyncBeforePublish &&
                     trial.surviving_weak_records > 0 && trial.has_strict_loss_gap);
                terminal_loss_control_tripped = terminal_loss_control_tripped ||
                    (spec.crash_frontier == CrashFrontier::TerminalUnfencedSuffix &&
                     trial.exposed_weak_records > 0 &&
                     trial.surviving_weak_records == 0 &&
                     trial.lost_weak_records == trial.exposed_weak_records);
                if (spec.crash_frontier == CrashFrontier::FenceAfterSyncBeforePublish &&
                    trial.post_recovery_suffix && trial.post_recovery_suffix_ok) {
                    ++post_recovery_suffixes;
                }
                if (spec.crash_frontier == CrashFrontier::StrictCapBeforeFence) {
                    const size_t strict_cap = static_cast<size_t>(
                        std::floor(spec.epsilon * static_cast<double>(options.k)));
                    const double cap_loss = static_cast<double>(strict_cap) /
                                            static_cast<double>(options.k);
                    const bool reached_cap_loss = std::any_of(
                        trial.crash.begin(), trial.crash.end(),
                        [&](const CrashObservation& observation) {
                            return std::abs(observation.membership_risk - cap_loss) <= 1e-12 &&
                                   std::abs(observation.realized_loss - cap_loss) <= 1e-12 &&
                                   std::abs(observation.positive_delta - cap_loss) <= 1e-12;
                        });
                    if (trial.exposed_weak_records == strict_cap &&
                        trial.surviving_weak_records == 0 &&
                        trial.lost_weak_records == strict_cap && reached_cap_loss) {
                        ++strict_cap_loss_images;
                    }
                }
                trials.push_back(std::move(trial));
            }
        }

        const auto raw_path = options.output_dir / "recall_committer_operations.csv";
        const auto crash_path = options.output_dir / "recall_committer_crash.csv";
        const auto aggregate_path = options.output_dir / "recall_committer.csv";
        writeRawCsv(raw_path, trials);
        writeCrashCsv(crash_path, trials);
        writeAggregateCsv(aggregate_path, trials);

        bool all_ok = negative_control_tripped && observed_graph_seeds.size() >= 2 &&
                      observed_strict_loss_gap && observed_partial_survival &&
                      terminal_loss_control_tripped &&
                      post_recovery_suffixes == options.repetitions &&
                      strict_cap_loss_images == options.repetitions;
        std::cout << "\nper-execution min/median/max (no query-row bootstrap)\n";
        for (const auto& spec : cases) {
            std::vector<double> throughputs;
            std::vector<double> deltas;
            for (const auto& trial : trials) {
                if (trial.case_name != spec.name) continue;
                throughputs.push_back(trial.throughput);
                double image_delta = 0.0;
                for (const auto& observation : trial.crash) {
                    image_delta += observation.positive_delta;
                }
                deltas.push_back(image_delta /
                                 static_cast<double>(std::max<size_t>(1, trial.crash.size())));
                all_ok = all_ok && trial.frontier_ok && trial.recovery_ok &&
                         trial.policy.cap_overshoots == 0;
                if (spec.policy == vdb::RecallPolicy::Strict) {
                    all_ok = all_ok && trial.strict_ok;
                }
            }
            std::cout << std::left << std::setw(20) << spec.name
                      << " writes/s=[" << *std::min_element(throughputs.begin(), throughputs.end())
                      << ',' << percentile(throughputs, .50) << ','
                      << *std::max_element(throughputs.begin(), throughputs.end())
                      << "] image_mean_Delta+=["
                      << *std::min_element(deltas.begin(), deltas.end()) << ','
                      << percentile(deltas, .50) << ','
                      << *std::max_element(deltas.begin(), deltas.end()) << "]\n";
        }
        std::cout << "raw_csv=" << raw_path << " crash_csv=" << crash_path
                  << " aggregate_csv=" << aggregate_path
                  << " changed_seed_control_tripped=" << (negative_control_tripped ? 1 : 0)
                  << " terminal_loss_control_tripped="
                  << (terminal_loss_control_tripped ? 1 : 0)
                  << " graph_seed_count=" << observed_graph_seeds.size()
                  << " partial_survival_observed="
                  << (observed_partial_survival ? 1 : 0)
                  << " post_recovery_suffixes=" << post_recovery_suffixes
                  << " strict_cap_loss_images=" << strict_cap_loss_images
                  << " observed_L_lt_M=" << (observed_strict_loss_gap ? 1 : 0)
                  << " invariants_ok=" << (all_ok ? 1 : 0) << '\n';
        if (!options.keep_images) std::filesystem::remove_all(root);
        return all_ok ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "recall-committer benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
