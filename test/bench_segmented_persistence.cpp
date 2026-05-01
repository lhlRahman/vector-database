#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

#include "../src/algorithms/flat_index.hpp"
#include "../src/algorithms/hnsw_index.hpp"
#include "../src/core/vector.hpp"
#include "../src/core/vector_accessor.hpp"
#include "../src/storage/mmap_storage.hpp"
#include "../src/storage/segmented_vector_store.hpp"
#include "../src/utils/distance_metrics.hpp"

using Clock = std::chrono::steady_clock;

struct VectorStore {
    std::vector<Vector> vectors;

    uint64_t add(Vector vector) {
        uint64_t id = vectors.size();
        vectors.push_back(std::move(vector));
        return id;
    }

    VectorAccessor accessor() {
        return [this](uint64_t id) -> const float* {
            return vectors[static_cast<size_t>(id)].data_ptr();
        };
    }
};

struct LatencyStats {
    double total_ms{0.0};
    double avg_us{0.0};
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double max_us{0.0};
};

struct BenchResult {
    std::string name;
    LatencyStats insert;
    LatencyStats update;
    LatencyStats erase;
    LatencyStats search;
    LatencyStats search_after_compact;
    double qps{0.0};
    double qps_after_compact{0.0};
    double recall_at_10{0.0};
    double recall_at_10_after_compact{0.0};
    double flush_ms{0.0};
    double seal_ms{0.0};
    double compact_ms{0.0};
    double recovery_ms{0.0};
    size_t live_vectors{0};
    size_t total_records{0};
    size_t tombstones{0};
    size_t segments{0};
    size_t sealed_segments{0};
    size_t disk_bytes{0};
    size_t wal_bytes{0};
    size_t vector_bytes{0};
    size_t hnsw_snapshot_bytes{0};
    size_t allocation_calls{0};
    size_t deallocation_calls{0};
    size_t peak_hnsw_bytes{0};
    double rss_delta_mib{0.0};
};

template <typename Fn>
double time_ms(Fn&& fn) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename Fn>
LatencyStats time_each(size_t count, Fn&& fn) {
    std::vector<double> samples;
    samples.reserve(count);

    auto total_start = Clock::now();
    for (size_t i = 0; i < count; ++i) {
        auto start = Clock::now();
        fn(i);
        auto end = Clock::now();
        samples.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    auto total_end = Clock::now();

    LatencyStats stats;
    stats.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    if (samples.empty()) return stats;

    std::sort(samples.begin(), samples.end());
    auto percentile = [&](double p) {
        size_t idx = static_cast<size_t>((p / 100.0) * static_cast<double>(samples.size() - 1));
        return samples[idx];
    };

    stats.avg_us = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    stats.p50_us = percentile(50.0);
    stats.p95_us = percentile(95.0);
    stats.p99_us = percentile(99.0);
    stats.max_us = samples.back();
    return stats;
}

Vector random_vec(size_t dims, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(dims);
    for (float& value : values) value = dist(rng);
    return Vector(values);
}

size_t recursive_size(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return 0;
    if (std::filesystem::is_regular_file(path, ec)) {
        auto size = std::filesystem::file_size(path, ec);
        return ec ? 0 : static_cast<size_t>(size);
    }

    size_t total = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        auto size = entry.file_size(ec);
        if (!ec) total += static_cast<size_t>(size);
    }
    return total;
}

size_t current_rss_bytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    size_t total_pages = 0;
    size_t rss_pages = 0;
    statm >> total_pages >> rss_pages;
    long page_size = sysconf(_SC_PAGESIZE);
    return static_cast<size_t>(rss_pages) * static_cast<size_t>(page_size);
#else
    return 0;
#endif
}

double mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double recall_at_k(const std::vector<std::vector<std::pair<std::string, float>>>& truth,
                   const std::vector<std::vector<std::pair<std::string, float>>>& actual,
                   size_t k) {
    size_t hits = 0;
    size_t total = 0;

    for (size_t i = 0; i < truth.size(); ++i) {
        size_t expected = std::min(k, truth[i].size());
        total += expected;
        for (size_t j = 0; j < expected; ++j) {
            const auto& expected_key = truth[i][j].first;
            auto found = std::find_if(actual[i].begin(), actual[i].end(), [&](const auto& pair) {
                return pair.first == expected_key;
            });
            if (found != actual[i].end()) ++hits;
        }
    }

    return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
}

std::vector<std::vector<std::pair<std::string, float>>>
exact_truth(size_t dims,
            const std::unordered_map<std::string, Vector>& live_vectors,
            const std::vector<Vector>& queries,
            size_t k) {
    VectorStore store;
    store.vectors.reserve(live_vectors.size());
    std::unordered_map<std::string, uint64_t> slots;
    slots.reserve(live_vectors.size());

    for (const auto& [key, vector] : live_vectors) {
        slots.emplace(key, store.add(vector));
    }

    FlatIndex<EuclideanMetricPolicy> exact(dims, store.accessor());
    std::vector<std::vector<std::pair<std::string, float>>> truth;
    truth.reserve(queries.size());
    for (const auto& query : queries) {
        truth.push_back(exact.search(query, k, slots));
    }
    return truth;
}

BenchResult run_monolithic(const std::filesystem::path& path,
                           const std::vector<Vector>& vectors,
                           const std::vector<Vector>& update_vectors,
                           const std::vector<size_t>& update_ids,
                           const std::vector<size_t>& delete_ids,
                           const std::vector<Vector>& queries,
                           const std::vector<std::vector<std::pair<std::string, float>>>& truth,
                           size_t dims,
                           size_t k,
                           size_t M,
                           size_t ef_construction,
                           size_t ef_search) {
    std::filesystem::remove(path);
    BenchResult result;
    result.name = "mmap_monolith";

    size_t rss_before = current_rss_bytes();
    std::vector<uint64_t> slots(vectors.size(), 0);
    std::unordered_map<std::string, uint64_t> live_slots;
    live_slots.reserve(vectors.size());

    MMapStorage storage(path.string(), dims, vectors.size() * 2);
    storage.open();

    auto accessor = [&](uint64_t slot_id) -> const float* {
        return storage.vector_ptr(slot_id);
    };

    auto metric = std::make_shared<EuclideanDistance>();
    HNSWIndex index(dims, M, ef_construction, ef_search, metric, accessor,
                    HNSWIndex::AllocationStrategy::Arena);

    result.insert = time_each(vectors.size(), [&](size_t i) {
        std::string key = "v" + std::to_string(i);
        uint64_t slot = storage.insert(key, vectors[i].data_ptr(), "meta_" + std::to_string(i));
        slots[i] = slot;
        live_slots[key] = slot;
        index.insert(slot, key);
    });

    result.erase = time_each(delete_ids.size(), [&](size_t i) {
        size_t id = delete_ids[i];
        std::string key = "v" + std::to_string(id);
        auto it = live_slots.find(key);
        if (it == live_slots.end()) return;
        storage.remove(it->second);
        index.remove(key);
        live_slots.erase(it);
    });

    result.update = time_each(update_ids.size(), [&](size_t i) {
        size_t id = update_ids[i];
        std::string key = "v" + std::to_string(id);
        auto it = live_slots.find(key);
        if (it == live_slots.end()) return;
        storage.update(it->second, update_vectors[i].data_ptr(), "updated_" + std::to_string(id));
        index.remove(key);
        index.insert(it->second, key);
    });

    std::vector<std::vector<std::pair<std::string, float>>> found;
    found.reserve(queries.size());
    result.search = time_each(queries.size(), [&](size_t i) {
        found.push_back(index.search(queries[i], k));
    });
    result.qps = result.search.total_ms == 0.0 ? 0.0
                 : static_cast<double>(queries.size()) * 1000.0 / result.search.total_ms;
    result.recall_at_10 = recall_at_k(truth, found, k);
    result.search_after_compact = result.search;
    result.qps_after_compact = result.qps;
    result.recall_at_10_after_compact = result.recall_at_10;

    result.flush_ms = time_ms([&] { storage.sync(); });
    auto mem = index.getMemoryStatistics();
    result.allocation_calls = mem.allocation_calls;
    result.deallocation_calls = mem.deallocation_calls;
    result.peak_hnsw_bytes = mem.peak_bytes_outstanding;
    result.live_vectors = live_slots.size();
    result.total_records = vectors.size();
    result.tombstones = delete_ids.size();
    result.segments = 1;
    result.sealed_segments = 0;
    result.disk_bytes = recursive_size(path);
    result.vector_bytes = result.disk_bytes;
    result.rss_delta_mib = mib(current_rss_bytes() - rss_before);

    storage.close();

    result.recovery_ms = time_ms([&] {
        MMapStorage recovered(path.string(), dims);
        recovered.open();
        auto recovered_slots = recovered.build_key_index();
        auto recovered_accessor = [&](uint64_t slot_id) -> const float* {
            return recovered.vector_ptr(slot_id);
        };
        HNSWIndex recovered_index(dims, M, ef_construction, ef_search, metric, recovered_accessor,
                                  HNSWIndex::AllocationStrategy::Arena);
        for (const auto& [key, slot] : recovered_slots) {
            recovered_index.insert(slot, key);
        }
        recovered.close();
    });

    return result;
}

BenchResult run_segmented(const std::filesystem::path& root,
                          const std::vector<Vector>& vectors,
                          const std::vector<Vector>& update_vectors,
                          const std::vector<size_t>& update_ids,
                          const std::vector<size_t>& delete_ids,
                          const std::vector<Vector>& queries,
                          const std::vector<std::vector<std::pair<std::string, float>>>& truth,
                          size_t dims,
                          size_t k,
                          size_t M,
                          size_t ef_construction,
                          size_t ef_search,
                          size_t max_mutable_records) {
    std::filesystem::remove_all(root);
    BenchResult result;
    result.name = "segmented";

    size_t rss_before = current_rss_bytes();

    SegmentedVectorStore::Config config;
    config.dimensions = dims;
    config.hnsw_m = M;
    config.hnsw_ef_construction = ef_construction;
    config.hnsw_ef_search = ef_search;
    config.max_mutable_segment_records = max_mutable_records;
    config.max_sealed_segments = 1024;
    config.max_tombstone_ratio = 1.0;
    config.allocation_strategy = HNSWIndex::AllocationStrategy::Arena;
    config.metric = std::make_shared<EuclideanDistance>();

    SegmentedVectorStore store(root, config);
    store.initialize();

    result.insert = time_each(vectors.size(), [&](size_t i) {
        store.insert(vectors[i], "v" + std::to_string(i), "meta_" + std::to_string(i));
    });

    result.erase = time_each(delete_ids.size(), [&](size_t i) {
        store.remove("v" + std::to_string(delete_ids[i]));
    });

    result.update = time_each(update_ids.size(), [&](size_t i) {
        size_t id = update_ids[i];
        store.update(update_vectors[i], "v" + std::to_string(id), "updated_" + std::to_string(id));
    });

    result.seal_ms = time_ms([&] {
        store.sealMutableSegment();
    });

    std::vector<std::vector<std::pair<std::string, float>>> found;
    found.reserve(queries.size());
    result.search = time_each(queries.size(), [&](size_t i) {
        found.push_back(store.search(queries[i], k));
    });
    result.qps = result.search.total_ms == 0.0 ? 0.0
                 : static_cast<double>(queries.size()) * 1000.0 / result.search.total_ms;
    result.recall_at_10 = recall_at_k(truth, found, k);

    result.flush_ms = time_ms([&] { store.flush(); });
    result.compact_ms = time_ms([&] { store.compact(); });

    std::vector<std::vector<std::pair<std::string, float>>> found_after_compact;
    found_after_compact.reserve(queries.size());
    result.search_after_compact = time_each(queries.size(), [&](size_t i) {
        found_after_compact.push_back(store.search(queries[i], k));
    });
    result.qps_after_compact = result.search_after_compact.total_ms == 0.0 ? 0.0
                               : static_cast<double>(queries.size()) * 1000.0 /
                                     result.search_after_compact.total_ms;
    result.recall_at_10_after_compact = recall_at_k(truth, found_after_compact, k);

    auto stats = store.getStatistics();
    result.live_vectors = stats.total_vectors;
    result.total_records = stats.total_records;
    result.tombstones = stats.total_tombstones;
    result.segments = stats.total_segments;
    result.sealed_segments = stats.sealed_segments;
    result.disk_bytes = stats.disk_bytes;
    result.wal_bytes = stats.wal_bytes;
    result.vector_bytes = stats.vector_bytes;
    result.hnsw_snapshot_bytes = stats.hnsw_snapshot_bytes;
    result.allocation_calls = stats.hnsw_allocation_calls;
    result.deallocation_calls = stats.hnsw_deallocation_calls;
    result.peak_hnsw_bytes = stats.hnsw_peak_bytes;
    result.rss_delta_mib = mib(current_rss_bytes() - rss_before);

    store.shutdown();

    result.recovery_ms = time_ms([&] {
        SegmentedVectorStore recovered(root, config);
        recovered.initialize();
        volatile size_t count = recovered.vectorCount();
        (void)count;
        recovered.shutdown();
    });

    return result;
}

void print_latency(const std::string& label, const LatencyStats& stats) {
    std::cout << std::left << std::setw(14) << label
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << stats.total_ms
              << std::setw(12) << stats.avg_us
              << std::setw(12) << stats.p50_us
              << std::setw(12) << stats.p95_us
              << std::setw(12) << stats.p99_us
              << std::setw(12) << stats.max_us << '\n';
}

void print_result(const BenchResult& result) {
    std::cout << "\n[" << result.name << "]\n";
    std::cout << std::left << std::setw(14) << "operation"
              << std::right << std::setw(12) << "total_ms"
              << std::setw(12) << "avg_us"
              << std::setw(12) << "p50_us"
              << std::setw(12) << "p95_us"
              << std::setw(12) << "p99_us"
              << std::setw(12) << "max_us" << '\n';
    std::cout << std::string(86, '-') << '\n';
    print_latency("insert", result.insert);
    print_latency("delete", result.erase);
    print_latency("update", result.update);
    print_latency("search", result.search);
    print_latency("search_post", result.search_after_compact);

    std::cout << "\n";
    std::cout << "qps=" << std::fixed << std::setprecision(0) << result.qps
              << " recall@10=" << std::setprecision(4) << result.recall_at_10
              << " post_qps=" << std::setprecision(0) << result.qps_after_compact
              << " post_recall@10=" << std::setprecision(4) << result.recall_at_10_after_compact
              << " flush_ms=" << std::setprecision(2) << result.flush_ms
              << " seal_ms=" << result.seal_ms
              << " compact_ms=" << result.compact_ms
              << " recovery_ms=" << result.recovery_ms << '\n';
    std::cout << "live=" << result.live_vectors
              << " records=" << result.total_records
              << " tombstones=" << result.tombstones
              << " segments=" << result.segments
              << " sealed=" << result.sealed_segments << '\n';
    std::cout << "disk_mib=" << std::setprecision(2) << mib(result.disk_bytes)
              << " wal_mib=" << mib(result.wal_bytes)
              << " vector_mib=" << mib(result.vector_bytes)
              << " hnsw_snapshot_mib=" << mib(result.hnsw_snapshot_bytes)
              << " rss_delta_mib=" << result.rss_delta_mib << '\n';
    std::cout << "hnsw_alloc_calls=" << result.allocation_calls
              << " hnsw_dealloc_calls=" << result.deallocation_calls
              << " hnsw_peak_mib=" << mib(result.peak_hnsw_bytes) << '\n';
}

int main(int argc, char** argv) {
    size_t n = 5000;
    size_t dims = 64;
    size_t query_count = 300;
    size_t k = 10;
    size_t M = 16;
    size_t ef_construction = 80;
    size_t ef_search = 50;

    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) dims = std::stoull(argv[2]);
    if (argc > 3) query_count = std::stoull(argv[3]);

    std::mt19937 rng(42);
    std::vector<Vector> vectors;
    vectors.reserve(n);
    for (size_t i = 0; i < n; ++i) vectors.push_back(random_vec(dims, rng));

    std::vector<Vector> queries;
    queries.reserve(query_count);
    for (size_t i = 0; i < query_count; ++i) queries.push_back(random_vec(dims, rng));

    std::vector<size_t> delete_ids;
    for (size_t i = 0; i < n; i += 20) delete_ids.push_back(i);

    std::unordered_set<size_t> deleted(delete_ids.begin(), delete_ids.end());
    std::vector<size_t> update_ids;
    for (size_t i = 1; i < n; i += 25) {
        if (deleted.count(i) == 0) update_ids.push_back(i);
    }

    std::vector<Vector> update_vectors;
    update_vectors.reserve(update_ids.size());
    for (size_t i = 0; i < update_ids.size(); ++i) update_vectors.push_back(random_vec(dims, rng));

    std::unordered_map<std::string, Vector> live_vectors;
    live_vectors.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (deleted.count(i) == 0) {
            live_vectors.emplace("v" + std::to_string(i), vectors[i]);
        }
    }
    for (size_t i = 0; i < update_ids.size(); ++i) {
        live_vectors["v" + std::to_string(update_ids[i])] = update_vectors[i];
    }

    auto truth = exact_truth(dims, live_vectors, queries, k);

    auto base = std::filesystem::temp_directory_path() /
                ("vdb_segment_bench_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(base);

    size_t max_mutable_records = std::max<size_t>(256, n / 4);

    std::cout << "Segmented persistence benchmark"
              << " n=" << n
              << " dims=" << dims
              << " queries=" << query_count
              << " delete_ops=" << delete_ids.size()
              << " update_ops=" << update_ids.size()
              << " k=" << k
              << " M=" << M
              << " efc=" << ef_construction
              << " efs=" << ef_search
              << " mutable_limit=" << max_mutable_records << "\n";

    BenchResult monolithic = run_monolithic(base / "legacy.vdb",
                                            vectors,
                                            update_vectors,
                                            update_ids,
                                            delete_ids,
                                            queries,
                                            truth,
                                            dims,
                                            k,
                                            M,
                                            ef_construction,
                                            ef_search);

    BenchResult segmented = run_segmented(base / "segmented",
                                          vectors,
                                          update_vectors,
                                          update_ids,
                                          delete_ids,
                                          queries,
                                          truth,
                                          dims,
                                          k,
                                          M,
                                          ef_construction,
                                          ef_search,
                                          max_mutable_records);

    print_result(monolithic);
    print_result(segmented);

    std::cout << "\n[segmented vs mmap_monolith]\n";
    std::cout << "insert_avg_speedup=" << monolithic.insert.avg_us / segmented.insert.avg_us
              << " search_avg_speedup=" << monolithic.search.avg_us / segmented.search.avg_us
              << " post_compact_search_speedup=" << monolithic.search.avg_us /
                     segmented.search_after_compact.avg_us
              << " recovery_speedup=" << monolithic.recovery_ms / segmented.recovery_ms
              << " disk_ratio=" << static_cast<double>(segmented.disk_bytes) /
                     static_cast<double>(std::max<size_t>(1, monolithic.disk_bytes))
              << " alloc_call_ratio=" << static_cast<double>(segmented.allocation_calls) /
                     static_cast<double>(std::max<size_t>(1, monolithic.allocation_calls))
              << '\n';

    std::filesystem::remove_all(base);
}
