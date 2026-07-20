#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../src/core/vector_database.hpp"
#include "../src/utils/atomic_write.hpp"
#include "../src/utils/vecs_io.hpp"

// End-to-end benchmark for the production recall-aware committer. Every trial
// uses the public API, clones a live database before shutdown, and opens that
// image through production read-only recovery. Exact truth comes from the
// externally recorded visible snapshot, never from recovered state.

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
    size_t repetitions{2};
    size_t ef{32};
    double strict_epsilon{0.20};
    double exchange_epsilon{0.05};
    uint32_t seed{100};
    std::string data_dir;
    std::filesystem::path output_dir{"build/ann_results"};
    bool keep_images{false};
};

struct DataSet {
    std::string name;
    size_t dimensions{0};
    std::vector<Values> base;
    std::vector<Values> queries;
    std::vector<Values> reserve;
};

enum class Workload { Random, Hot };

struct CaseSpec {
    std::string name;
    Workload workload{Workload::Random};
    vdb::RecallPolicy policy{vdb::RecallPolicy::Strict};
    vdb::AckMode ack{vdb::AckMode::Weak};
    double epsilon{0.0};
    size_t record_cap{std::numeric_limits<size_t>::max()};
    std::chrono::milliseconds age_cap{0};
    bool correlation_guard{false};
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
    bool durable_fingerprint_equal{true};
};

struct TrialResult {
    std::string case_name;
    std::string workload;
    size_t repetition{0};
    uint32_t seed{0};
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
    vdb::RecallCommitPolicyCounters policy;
    std::vector<CrashObservation> crash;
    size_t stable_losses{0};
    size_t unexpected_weak_survivors{0};
    bool frontier_ok{true};
    bool strict_ok{true};
    bool recovery_ok{true};
    bool alarmed{false};
    double alarm_latency_ms{-1.0};
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

std::pair<double, double> bootstrapMeanCI(const std::vector<double>& samples,
                                          uint32_t seed) {
    if (samples.empty()) return {0.0, 0.0};
    std::mt19937 generator(seed);
    std::uniform_int_distribution<size_t> pick(0, samples.size() - 1);
    std::vector<double> means;
    means.reserve(1000);
    for (size_t replicate = 0; replicate < 1000; ++replicate) {
        double sum = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) sum += samples[pick(generator)];
        means.push_back(sum / static_cast<double>(samples.size()));
    }
    return {percentile(means, 0.025), percentile(means, 0.975)};
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
                                   size_t k) {
    std::vector<std::pair<double, std::string>> candidates;
    candidates.reserve(data.base.size() + tail.size());
    for (size_t i = 0; i < data.base.size(); ++i) {
        candidates.emplace_back(l2(query, data.base[i]), "base-" + std::to_string(i));
    }
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& receipt = operations[i].receipt;
        if (receipt.applied && receipt.lsn <= snapshot_lsn) {
            candidates.emplace_back(l2(query, tail[i]), "tail-" + std::to_string(i));
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
                                     bool reverse_order) {
    const auto path = root / (reverse_order ? "changed-seed-base" : "base");
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
                                const std::filesystem::path& production_base) {
    const auto changed = buildBaseImage(data, options, root, options.seed + 1, true);
    VectorDatabase production(data.dimensions, VectorDatabase::SearchMode::HNSW,
                              false, false, {}, false, 0, production_base.string(),
                              VectorDatabase::StorageEngine::Segmented,
                              vdb::OpenMode::ReadOnlyRecovery);
    VectorDatabase control(data.dimensions, VectorDatabase::SearchMode::HNSW,
                           false, false, {}, false, 0, changed.string(),
                           VectorDatabase::StorageEngine::Segmented,
                           vdb::OpenMode::ReadOnlyRecovery);
    production.configureHNSW(16, 100, options.ef, options.seed);
    control.configureHNSW(16, 100, options.ef, options.seed + 1);
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
    const bool topology_fingerprint_equal = persistedSeed(changed) == options.seed;
    const bool accepted = topology_fingerprint_equal && !answer_drift;
    production.shutdown();
    control.shutdown();
    std::cout << "negative_control changed_seed=" << options.seed + 1
              << " answer_drift=" << (answer_drift ? 1 : 0)
              << " topology_fingerprint_mismatch="
              << (!topology_fingerprint_equal ? 1 : 0)
              << " validator_rejected=" << (!accepted ? 1 : 0) << '\n';
    return !accepted;
}

TrialResult runTrial(const DataSet& data,
                     const Options& options,
                     const CaseSpec& spec,
                     size_t repetition,
                     const std::filesystem::path& base_image,
                     const std::filesystem::path& root) {
    TrialResult trial;
    trial.case_name = spec.name;
    trial.workload = workloadName(spec.workload);
    trial.repetition = repetition;
    trial.seed = options.seed;
    trial.writer_count = options.writers;
    const auto database_path = root / (spec.name + "-" + std::to_string(repetition));
    const auto crash_path = root / (spec.name + "-" + std::to_string(repetition) + "-crash");
    copyTree(base_image, database_path);
    const auto config = makeConfig(spec, options, options.seed);
    const auto tail = makeTail(data, options, spec.workload, repetition);

    VectorDatabase database(data.dimensions, VectorDatabase::SearchMode::HNSW,
                            false, false, {}, false, 0, database_path.string());
    database.configureHNSW(16, 100, options.ef, options.seed);
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
    query_thread.join();
    if (error) std::rethrow_exception(error);
    trial.write_seconds = std::chrono::duration<double>(writers_end - workload_start).count();
    trial.throughput = static_cast<double>(options.writes) / trial.write_seconds;

    auto pre_status = database.durabilityStatus();
    if (pre_status.correlation_alarm && pre_status.weak_records != 0) {
        (void)database.waitUntilDurable(pre_status.visible_lsn, std::chrono::seconds(2));
        pre_status = database.durabilityStatus();
    }
    copyTree(database_path, crash_path);

    VectorDatabase recovered(data.dimensions, VectorDatabase::SearchMode::HNSW,
                             false, false, {}, false, 0, crash_path.string(),
                             VectorDatabase::StorageEngine::Segmented,
                             vdb::OpenMode::ReadOnlyRecovery);
    recovered.configureHNSW(16, 100, options.ef, options.seed);
    recovered.configureRecallCommit(config);
    recovered.initialize();

    std::unordered_set<std::string> weak_at_crash;
    for (const auto& operation : trial.writes) {
        if (operation.receipt.lsn > pre_status.durable_lsn) {
            weak_at_crash.insert("tail-" + std::to_string(operation.index));
        }
    }
    for (size_t i = 0; i < data.base.size(); ++i) {
        if (!recovered.inspectRecord("base-" + std::to_string(i))) ++trial.stable_losses;
    }
    for (const auto& operation : trial.writes) {
        const std::string key = "tail-" + std::to_string(operation.index);
        const bool present = recovered.inspectRecord(key).has_value();
        if (operation.receipt.lsn <= pre_status.durable_lsn && !present) ++trial.stable_losses;
        if (operation.receipt.lsn > pre_status.durable_lsn && present) {
            ++trial.unexpected_weak_survivors;
        }
    }

    const size_t crash_queries = std::min<size_t>(12, data.queries.size());
    for (size_t q = 0; q < crash_queries; ++q) {
        const size_t query_index = spec.workload == Workload::Hot
                                       ? q % std::min<size_t>(4, data.queries.size())
                                       : q;
        const Vector query(data.queries[query_index]);
        const auto pre = database.similaritySearch(query, options.k,
                                                   vdb::ReadVisibility::Latest);
        const auto stable = database.similaritySearch(query, options.k,
                                                      vdb::ReadVisibility::Stable);
        const auto post = recovered.similaritySearch(query, options.k,
                                                      vdb::ReadVisibility::Latest);
        const auto truth = exactTopK(data.queries[query_index], data, tail,
                                     trial.writes, pre.snapshot_lsn, options.k);
        const auto pre_keys = responseKeys(pre);
        const auto post_keys = responseKeys(post);
        const auto stable_keys = responseKeys(stable);
        size_t weak_truth = 0;
        for (const auto& key : truth) weak_truth += weak_at_crash.contains(key);
        const double denominator = static_cast<double>(std::max<size_t>(1, truth.size()));
        const double pre_recall = overlap(pre_keys, truth, truth.size());
        const double post_recall = overlap(post_keys, truth, truth.size());
        bool subset = true;
        const std::unordered_set<std::string> post_set(post_keys.begin(), post_keys.end());
        for (const auto& key : pre_keys) {
            if (!post_set.contains(key) && !weak_at_crash.contains(key)) subset = false;
        }
        CrashObservation observation;
        observation.membership_risk = static_cast<double>(weak_truth) / denominator;
        observation.realized_loss = observation.membership_risk;
        observation.positive_delta = std::max(0.0, pre_recall - post_recall);
        observation.amplification = std::max(
            0.0, observation.positive_delta - observation.realized_loss);
        observation.pre_recall = pre_recall;
        observation.post_recall = post_recall;
        observation.answer_churn = 1.0 - overlap(pre_keys, post_keys, options.k);
        observation.durable_overlap = overlap(stable_keys, post_keys, options.k);
        observation.lost_ids_subset_weak = subset;
        observation.durable_fingerprint_equal =
            keyFingerprint(stable_keys) == keyFingerprint(post_keys);
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
    trial.policy = database.recallPolicyStatistics();
    trial.alarmed = trial.policy.correlation.alarmed;
    if (alarm_us.load() >= 0) trial.alarm_latency_ms = alarm_us.load() / 1000.0;

    const auto fence_begin = Clock::now();
    (void)database.durabilityFence();
    trial.fence_latency_us = elapsedUs(fence_begin, Clock::now());
    trial.committer = database.recallCommitterStatistics();
    trial.frontier_ok = trial.frontier_ok && pre_status.appended_lsn >= pre_status.visible_lsn &&
                        pre_status.visible_lsn >= pre_status.durable_lsn;
    trial.recovery_ok = trial.stable_losses == 0 && trial.unexpected_weak_survivors == 0;
    for (const auto& observation : trial.crash) {
        trial.recovery_ok = trial.recovery_ok && observation.lost_ids_subset_weak &&
                            observation.durable_fingerprint_equal &&
                            observation.amplification <= 1e-12;
        if (spec.policy == vdb::RecallPolicy::Strict) {
            trial.strict_ok = trial.strict_ok &&
                observation.positive_delta <= observation.realized_loss + 1e-12 &&
                observation.realized_loss <= observation.membership_risk + 1e-12 &&
                observation.membership_risk <= spec.epsilon + 1e-12;
        }
    }
    trial.strict_ok = trial.strict_ok && trial.policy.cap_overshoots == 0;
    recovered.shutdown();
    database.shutdown();
    if (!options.keep_images) {
        std::filesystem::remove_all(database_path);
        std::filesystem::remove_all(crash_path);
    }
    return trial;
}

void writeRawCsv(const std::filesystem::path& path,
                 const std::vector<TrialResult>& trials) {
    std::ofstream output(path);
    output << "case,workload,repetition,seed,operation,index,ack,latency_us,lsn,"
              "visible_lsn,durable_lsn,durable_records,weak_records,cap,risk,"
              "snapshot_lsn,exact_recall,tail_evaluations\n";
    for (const auto& trial : trials) {
        for (const auto& operation : trial.writes) {
            const auto& receipt = operation.receipt;
            output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                   << ',' << trial.seed << ",write," << operation.index << ','
                   << ackName(receipt.actual_ack) << ',' << operation.latency_us << ','
                   << receipt.lsn << ',' << receipt.visible_lsn << ','
                   << receipt.durable_lsn << ',' << receipt.durable_count << ','
                   << receipt.weak_count << ',' << receipt.policy_cap << ','
                   << receipt.risk_estimate << ",,,\n";
        }
        for (size_t i = 0; i < trial.queries.size(); ++i) {
            const auto& query = trial.queries[i];
            output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
                   << ',' << trial.seed << ",query," << i << ",," << query.latency_us
                   << ",," << query.response.snapshot_lsn << ','
                   << query.response.durable_lsn << ',' << query.status.durable_records
                   << ',' << query.status.weak_records << ',' << query.status.policy_record_cap
                   << ',' << query.status.estimated_recall_loss << ','
                   << query.response.snapshot_lsn << ',' << query.exact_recall << ','
                   << query.response.exact_tail_distance_evaluations << '\n';
        }
        output << trial.case_name << ',' << trial.workload << ',' << trial.repetition
               << ',' << trial.seed << ",fence,0,stable," << trial.fence_latency_us
               << ",,,,,,,,,,\n";
    }
}

void writeAggregateCsv(const std::filesystem::path& path,
                       const std::vector<TrialResult>& trials) {
    std::ofstream output(path);
    output << "case,workload,repetition,writers,writes_per_s,weak_p50_us,weak_p95_us,"
              "weak_p99_us,stable_p50_us,stable_p95_us,stable_p99_us,fence_us,"
              "query_p50_us,query_p95_us,query_p99_us,mean_D,mean_W,max_W,max_cap,"
              "max_risk,sync_attempts,sync_successes,sync_failures,syncs_per_s,records_per_sync,"
              "policy_fences,age_fences,overshoots,enrichment,alarm,alarm_latency_ms,"
              "M_max,L_max,delta_max,amplification_max,stable_losses,weak_survivors,"
              "frontier_ok,recovery_ok,strict_ok\n";
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
               << ',' << trial.max_risk << ',' << trial.committer.sync_attempts << ','
               << trial.committer.sync_successes << ',' << trial.committer.sync_failures
               << ',' << trial.committer.sync_successes / trial.write_seconds
               << ',' << records_per_sync << ',' << trial.committer.policy_fences << ','
               << trial.committer.age_fences << ',' << trial.policy.cap_overshoots << ','
               << trial.policy.correlation.enrichment << ',' << (trial.alarmed ? 1 : 0)
               << ',' << trial.alarm_latency_ms << ',' << m << ',' << l << ',' << delta
               << ',' << amplification << ',' << trial.stable_losses << ','
               << trial.unexpected_weak_survivors << ',' << (trial.frontier_ok ? 1 : 0)
               << ',' << (trial.recovery_ok ? 1 : 0) << ',' << (trial.strict_ok ? 1 : 0)
               << '\n';
    }
}

void writeCrashCsv(const std::filesystem::path& path,
                   const std::vector<TrialResult>& trials) {
    std::ofstream output(path);
    output << "case,workload,repetition,query,M,L,delta_positive,amplification,"
              "pre_recall,post_recall,answer_churn,durable_overlap,"
              "lost_ids_subset_weak,durable_fingerprint_equal\n";
    for (const auto& trial : trials) {
        for (size_t i = 0; i < trial.crash.size(); ++i) {
            const auto& observation = trial.crash[i];
            output << trial.case_name << ',' << trial.workload << ','
                   << trial.repetition << ',' << i << ','
                   << observation.membership_risk << ',' << observation.realized_loss
                   << ',' << observation.positive_delta << ','
                   << observation.amplification << ',' << observation.pre_recall << ','
                   << observation.post_recall << ',' << observation.answer_churn << ','
                   << observation.durable_overlap << ','
                   << (observation.lost_ids_subset_weak ? 1 : 0) << ','
                   << (observation.durable_fingerprint_equal ? 1 : 0) << '\n';
        }
    }
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
        else if (argument == "--repetitions" || argument == "--seeds") options.repetitions = std::stoul(next());
        else if (argument == "--ef") options.ef = std::stoul(next());
        else if (argument == "--epsilon") options.strict_epsilon = std::stod(next());
        else if (argument == "--exchange-epsilon") options.exchange_epsilon = std::stod(next());
        else if (argument == "--output") options.output_dir = next();
        else if (argument == "--keep-images") options.keep_images = true;
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.writers == 0 || options.writes == 0 || options.queries == 0 ||
        options.base_records == 0 || options.k == 0 || options.repetitions == 0) {
        throw std::invalid_argument("benchmark counts must be nonzero");
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

        const auto base = buildBaseImage(data, options, root, options.seed, false);
        const bool negative_control_tripped =
            validateChangedSeedControl(data, options, root, base);
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
             vdb::AckMode::Weak, options.strict_epsilon},
            {"strict-hot", Workload::Hot, vdb::RecallPolicy::Strict,
             vdb::AckMode::Weak, options.strict_epsilon},
            {"exchange-random", Workload::Random, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, options.exchange_epsilon},
            {"exchange-hot", Workload::Hot, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, options.exchange_epsilon},
            {"exchange-hot-guard", Workload::Hot, vdb::RecallPolicy::ExchangeableMean,
             vdb::AckMode::Weak, options.exchange_epsilon,
             std::numeric_limits<size_t>::max(), std::chrono::milliseconds(0), true},
        };

        std::vector<TrialResult> trials;
        for (const auto& spec : cases) {
            for (size_t repetition = 0; repetition < options.repetitions; ++repetition) {
                auto trial = runTrial(data, options, spec, repetition, base, root);
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
                          << " amp=" << max_amp << " syncs="
                          << trial.committer.sync_successes << " alarm="
                          << (trial.alarmed ? 1 : 0) << " strict_ok="
                          << (trial.strict_ok ? 1 : 0) << " recovery_ok="
                          << (trial.recovery_ok ? 1 : 0) << '\n';
                trials.push_back(std::move(trial));
            }
        }

        const auto raw_path = options.output_dir / "recall_commit_operations.csv";
        const auto crash_path = options.output_dir / "recall_commit_crash.csv";
        const auto aggregate_path = options.output_dir / "recall_commit.csv";
        writeRawCsv(raw_path, trials);
        writeCrashCsv(crash_path, trials);
        writeAggregateCsv(aggregate_path, trials);

        bool all_ok = negative_control_tripped;
        std::cout << "\nbootstrap 95% confidence intervals (1000 deterministic resamples)\n";
        for (const auto& spec : cases) {
            std::vector<double> throughputs;
            std::vector<double> deltas;
            for (const auto& trial : trials) {
                if (trial.case_name != spec.name) continue;
                throughputs.push_back(trial.throughput);
                for (const auto& observation : trial.crash) {
                    deltas.push_back(observation.positive_delta);
                }
                all_ok = all_ok && trial.frontier_ok && trial.recovery_ok &&
                         trial.policy.cap_overshoots == 0;
                if (spec.policy == vdb::RecallPolicy::Strict) {
                    all_ok = all_ok && trial.strict_ok;
                }
            }
            const auto throughput_ci = bootstrapMeanCI(throughputs, 10 + options.seed);
            const auto delta_ci = bootstrapMeanCI(deltas, 20 + options.seed);
            std::cout << std::left << std::setw(20) << spec.name
                      << " writes/s_ci=[" << throughput_ci.first << ','
                      << throughput_ci.second << "] Delta+_ci=[" << delta_ci.first
                      << ',' << delta_ci.second << "]\n";
        }
        std::cout << "raw_csv=" << raw_path << " crash_csv=" << crash_path
                  << " aggregate_csv=" << aggregate_path
                  << " changed_seed_control_tripped=" << (negative_control_tripped ? 1 : 0)
                  << " invariants_ok=" << (all_ok ? 1 : 0) << '\n';
        if (!options.keep_images) std::filesystem::remove_all(root);
        return all_ok ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "recall-committer benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
