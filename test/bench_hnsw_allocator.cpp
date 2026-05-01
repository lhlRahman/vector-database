#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/algorithms/flat_index.hpp"
#include "../src/algorithms/hnsw_index.hpp"
#include "../src/core/vector.hpp"
#include "../src/core/vector_accessor.hpp"
#include "../src/utils/distance_metrics.hpp"

using Clock = std::chrono::steady_clock;

struct VectorStore {
    std::vector<Vector> vectors;

    uint64_t add(Vector v) {
        uint64_t id = vectors.size();
        vectors.push_back(std::move(v));
        return id;
    }

    VectorAccessor accessor() {
        return [this](uint64_t id) -> const float* {
            return vectors[id].data_ptr();
        };
    }
};

struct Result {
    std::string name;
    double build_ms;
    double search_ms;
    double avg_query_us;
    double qps;
    double recall_at_10;
    HNSWIndex::MemoryStatistics memory;
};

static Vector random_vec(size_t dims, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(dims);
    for (float& value : values) value = dist(rng);
    return Vector(values);
}

template <typename Fn>
static double time_ms(Fn&& fn) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double recall_at_k(const std::vector<std::vector<std::pair<std::string, float>>>& truth,
                          const std::vector<std::vector<std::pair<std::string, float>>>& actual,
                          size_t k) {
    size_t hits = 0;
    size_t total = 0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const size_t expected = std::min(k, truth[i].size());
        total += expected;
        for (size_t j = 0; j < expected; ++j) {
            const std::string& key = truth[i][j].first;
            auto found = std::find_if(actual[i].begin(), actual[i].end(),
                                      [&](const auto& pair) { return pair.first == key; });
            if (found != actual[i].end()) ++hits;
        }
    }
    return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
}

static Result run_case(const std::string& name,
                       HNSWIndex::AllocationStrategy strategy,
                       VectorStore& store,
                       const std::unordered_map<std::string, uint64_t>& slots,
                       const std::vector<Vector>& queries,
                       const std::vector<std::vector<std::pair<std::string, float>>>& truth,
                       size_t dims,
                       size_t k,
                       size_t M,
                       size_t ef_construction,
                       size_t ef_search) {
    auto metric = std::make_shared<EuclideanDistance>();
    HNSWIndex index(dims, M, ef_construction, ef_search, metric, store.accessor(), strategy);

    double build_ms = time_ms([&] {
        for (const auto& [key, slot_id] : slots) {
            index.insert(slot_id, key);
        }
    });

    std::vector<std::vector<std::pair<std::string, float>>> results;
    results.reserve(queries.size());
    double search_ms = time_ms([&] {
        for (const auto& query : queries) {
            results.push_back(index.search(query, k));
        }
    });

    return Result{
        name,
        build_ms,
        search_ms,
        (search_ms * 1000.0) / static_cast<double>(queries.size()),
        static_cast<double>(queries.size()) * 1000.0 / search_ms,
        recall_at_k(truth, results, k),
        index.getMemoryStatistics(),
    };
}

static void print_result(const Result& result) {
    const double mib = static_cast<double>(result.memory.peak_bytes_outstanding) / (1024.0 * 1024.0);
    std::cout << std::left << std::setw(12) << result.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << result.build_ms
              << std::setw(12) << result.search_ms
              << std::setw(12) << result.avg_query_us
              << std::setw(12) << std::setprecision(0) << result.qps
              << std::setw(12) << std::setprecision(3) << result.recall_at_10
              << std::setw(14) << result.memory.allocation_calls
              << std::setw(14) << std::setprecision(2) << mib
              << "\n";
}

int main(int argc, char** argv) {
    size_t n = 5000;
    size_t dims = 64;
    size_t queries_count = 500;
    size_t k = 10;
    size_t M = 16;
    size_t ef_construction = 80;
    size_t ef_search = 50;

    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) dims = std::stoull(argv[2]);
    if (argc > 3) queries_count = std::stoull(argv[3]);

    std::mt19937 rng(42);
    VectorStore store;
    store.vectors.reserve(n);
    std::unordered_map<std::string, uint64_t> slots;
    slots.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        std::string key = "v" + std::to_string(i);
        slots.emplace(key, store.add(random_vec(dims, rng)));
    }

    std::vector<Vector> queries;
    queries.reserve(queries_count);
    for (size_t i = 0; i < queries_count; ++i) {
        queries.push_back(random_vec(dims, rng));
    }

    FlatIndex<EuclideanMetricPolicy> exact(dims, store.accessor());
    std::vector<std::vector<std::pair<std::string, float>>> truth;
    truth.reserve(queries.size());
    for (const auto& query : queries) {
        truth.push_back(exact.search(query, k, slots));
    }

    Result standard = run_case("standard",
                               HNSWIndex::AllocationStrategy::Standard,
                               store, slots, queries, truth, dims, k,
                               M, ef_construction, ef_search);
    Result arena = run_case("arena",
                            HNSWIndex::AllocationStrategy::Arena,
                            store, slots, queries, truth, dims, k,
                            M, ef_construction, ef_search);

    std::cout << "HNSW allocator benchmark"
              << "  n=" << n
              << " dims=" << dims
              << " queries=" << queries_count
              << " k=" << k
              << " M=" << M
              << " efc=" << ef_construction
              << " efs=" << ef_search << "\n\n";

    std::cout << std::left << std::setw(12) << "allocator"
              << std::right << std::setw(12) << "build_ms"
              << std::setw(12) << "search_ms"
              << std::setw(12) << "avg_us"
              << std::setw(12) << "qps"
              << std::setw(12) << "recall@10"
              << std::setw(14) << "alloc_calls"
              << std::setw(14) << "peak_mib"
              << "\n";
    std::cout << std::string(100, '-') << "\n";
    print_result(standard);
    print_result(arena);

    std::cout << "\nArena vs standard:\n"
              << "  build speedup:  " << std::setprecision(2) << standard.build_ms / arena.build_ms << "x\n"
              << "  search speedup: " << standard.avg_query_us / arena.avg_query_us << "x\n"
              << "  alloc call reduction: "
              << static_cast<double>(standard.memory.allocation_calls) /
                     static_cast<double>(std::max<size_t>(1, arena.memory.allocation_calls))
              << "x\n";
}
