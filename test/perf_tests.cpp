// test/perf_tests.cpp
// Performance / benchmark tests for the vector database

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../src/core/vector.hpp"
#include "../src/core/vector_database.hpp"
#include "../src/utils/distance_metrics.hpp"

// ---- timing helpers ----

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    std::string name;
    double total_ms;
    size_t ops;
    double ops_per_sec;
    double avg_us; // microseconds per op
};

static BenchResult bench(const std::string& name, size_t ops, std::function<void()> fn) {
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_sec = (ms > 0) ? (static_cast<double>(ops) / ms * 1000.0) : 0;
    double avg_us = (ops > 0) ? (ms * 1000.0 / static_cast<double>(ops)) : 0;
    return {name, ms, ops, per_sec, avg_us};
}

static void print_result(const BenchResult& r) {
    std::cout << "  " << std::left << std::setw(40) << r.name
              << std::right << std::setw(10) << std::fixed << std::setprecision(1) << r.total_ms << " ms"
              << std::setw(12) << static_cast<size_t>(r.ops_per_sec) << " ops/s"
              << std::setw(10) << std::setprecision(1) << r.avg_us << " us/op"
              << "\n";
}

// ---- vector generation ----

static Vector random_vec(size_t dims, unsigned seed) {
    std::vector<float> v(dims);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dims; i++) v[i] = dist(rng);
    return Vector(v);
}

// =====================================================================
//  BENCHMARK: Single insert throughput
// =====================================================================

BenchResult bench_single_insert(size_t num_vectors, size_t dims) {
    VectorDatabase db(dims);
    db.initialize();

    return bench("insert " + std::to_string(num_vectors) + " vectors (" + std::to_string(dims) + "D)",
                 num_vectors, [&]() {
        for (size_t i = 0; i < num_vectors; i++) {
            (void)db.insert(random_vec(dims, static_cast<unsigned>(i)), "v" + std::to_string(i));
        }
    });
}

// =====================================================================
//  BENCHMARK: Batch insert throughput
// =====================================================================

BenchResult bench_batch_insert(size_t num_vectors, size_t dims) {
    VectorDatabase db(dims, "exact", false, true);
    db.initialize();

    std::vector<std::string> keys;
    std::vector<Vector> vectors;
    keys.reserve(num_vectors);
    vectors.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; i++) {
        keys.push_back("b" + std::to_string(i));
        vectors.push_back(random_vec(dims, static_cast<unsigned>(i)));
    }

    return bench("batch insert " + std::to_string(num_vectors) + " vectors (" + std::to_string(dims) + "D)",
                 num_vectors, [&]() {
        (void)db.batchInsert(keys, vectors);
    });
}

// =====================================================================
//  BENCHMARK: Search latency (exact, LSH, HNSW)
// =====================================================================

BenchResult bench_search(const std::string& algorithm, size_t db_size, size_t dims,
                         size_t k, size_t num_queries) {
    VectorDatabase db(dims, algorithm);
    db.initialize();

    for (size_t i = 0; i < db_size; i++) {
        (void)db.insert(random_vec(dims, static_cast<unsigned>(i)), "v" + std::to_string(i));
    }

    // Pre-generate queries
    std::vector<Vector> queries;
    queries.reserve(num_queries);
    for (size_t i = 0; i < num_queries; i++) {
        queries.push_back(random_vec(dims, static_cast<unsigned>(i + db_size + 1000)));
    }

    std::string label = algorithm + " search (n=" + std::to_string(db_size) +
                        ", d=" + std::to_string(dims) + ", k=" + std::to_string(k) + ")";

    return bench(label, num_queries, [&]() {
        for (size_t i = 0; i < num_queries; i++) {
            auto r = db.similaritySearch(queries[i], k);
            (void)r;
        }
    });
}

// =====================================================================
//  BENCHMARK: Update throughput
// =====================================================================

BenchResult bench_update(size_t db_size, size_t num_updates, size_t dims) {
    VectorDatabase db(dims);
    db.initialize();

    for (size_t i = 0; i < db_size; i++) {
        (void)db.insert(random_vec(dims, static_cast<unsigned>(i)), "v" + std::to_string(i));
    }

    return bench("update " + std::to_string(num_updates) + " vectors (n=" + std::to_string(db_size) + ")",
                 num_updates, [&]() {
        for (size_t i = 0; i < num_updates; i++) {
            (void)db.update(random_vec(dims, static_cast<unsigned>(i + 50000)),
                            "v" + std::to_string(i % db_size));
        }
    });
}

// =====================================================================
//  BENCHMARK: Delete throughput
// =====================================================================

BenchResult bench_delete(size_t db_size, size_t dims) {
    VectorDatabase db(dims);
    db.initialize();

    for (size_t i = 0; i < db_size; i++) {
        (void)db.insert(random_vec(dims, static_cast<unsigned>(i)), "v" + std::to_string(i));
    }

    return bench("delete " + std::to_string(db_size) + " vectors", db_size, [&]() {
        for (size_t i = 0; i < db_size; i++) {
            (void)db.remove("v" + std::to_string(i));
        }
    });
}

// =====================================================================
//  BENCHMARK: Cache hit vs miss
// =====================================================================

std::pair<BenchResult, BenchResult> bench_cache(size_t db_size, size_t dims, size_t num_queries) {
    // Cache enabled
    VectorDatabase db_cached(dims, "exact", false, false, {}, true, 1000);
    db_cached.initialize();

    for (size_t i = 0; i < db_size; i++) {
        (void)db_cached.insert(random_vec(dims, static_cast<unsigned>(i)), "v" + std::to_string(i));
    }

    Vector query = random_vec(dims, 99999);

    // Warm the cache with first query
    (void)db_cached.similaritySearch(query, 10);

    auto cache_hit = bench("search (cache hit, n=" + std::to_string(db_size) + ")", num_queries, [&]() {
        for (size_t i = 0; i < num_queries; i++) {
            (void)db_cached.similaritySearch(query, 10);
        }
    });

    // Cache disabled
    VectorDatabase db_no_cache(dims, "exact", false, false, {}, false);
    db_no_cache.initialize();

    for (size_t i = 0; i < db_size; i++) {
        (void)db_no_cache.insert(random_vec(dims, static_cast<unsigned>(i)), "v" + std::to_string(i));
    }

    auto cache_miss = bench("search (no cache, n=" + std::to_string(db_size) + ")", num_queries, [&]() {
        for (size_t i = 0; i < num_queries; i++) {
            (void)db_no_cache.similaritySearch(query, 10);
        }
    });

    return {cache_hit, cache_miss};
}

// =====================================================================
//  BENCHMARK: Concurrent read throughput
// =====================================================================

BenchResult bench_concurrent_reads(size_t db_size, size_t dims, int num_threads, size_t queries_per_thread) {
    VectorDatabase db(dims);
    db.initialize();

    for (size_t i = 0; i < db_size; i++) {
        (void)db.insert(random_vec(dims, static_cast<unsigned>(i)), "v" + std::to_string(i));
    }

    size_t total_queries = static_cast<size_t>(num_threads) * queries_per_thread;

    return bench("concurrent search (" + std::to_string(num_threads) + " threads, n=" +
                 std::to_string(db_size) + ")", total_queries, [&]() {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([&, t]() {
                for (size_t q = 0; q < queries_per_thread; q++) {
                    auto seed = static_cast<unsigned>(t * 10000 + q);
                    (void)db.similaritySearch(random_vec(dims, seed), 10);
                }
            });
        }
        for (auto& t : threads) t.join();
    });
}

// =====================================================================
//  BENCHMARK: Scaling with dimensions
// =====================================================================

void bench_dimension_scaling(size_t db_size, size_t num_queries) {
    std::cout << "\n  Dimension scaling (n=" << db_size << ", " << num_queries << " queries):\n";

    for (size_t dims : {8, 32, 64, 128, 256}) {
        auto r = bench_search("exact", db_size, dims, 10, num_queries);
        std::cout << "    d=" << std::setw(4) << dims << ": "
                  << std::fixed << std::setprecision(1) << r.avg_us << " us/query"
                  << " (" << std::setprecision(0) << r.ops_per_sec << " qps)\n";
    }
}

// =====================================================================
//  BENCHMARK: Scaling with database size
// =====================================================================

void bench_size_scaling(size_t dims, size_t num_queries) {
    std::cout << "\n  Size scaling (d=" << dims << ", " << num_queries << " queries):\n";

    for (size_t n : {100, 500, 1000, 5000}) {
        auto r = bench_search("exact", n, dims, 10, num_queries);
        std::cout << "    n=" << std::setw(5) << n << ": "
                  << std::fixed << std::setprecision(1) << r.avg_us << " us/query"
                  << " (" << std::setprecision(0) << r.ops_per_sec << " qps)\n";
    }
}

// =====================================================================
//  MAIN
// =====================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " Vector Database Performance Tests\n";
    std::cout << "========================================\n\n";

    std::vector<BenchResult> results;

    // -- Insert throughput --
    std::cout << "[Insert Throughput]\n";
    results.push_back(bench_single_insert(1000, 32));
    print_result(results.back());
    results.push_back(bench_single_insert(5000, 32));
    print_result(results.back());
    results.push_back(bench_single_insert(1000, 128));
    print_result(results.back());

    // -- Batch insert --
    std::cout << "\n[Batch Insert Throughput]\n";
    results.push_back(bench_batch_insert(1000, 32));
    print_result(results.back());
    results.push_back(bench_batch_insert(5000, 32));
    print_result(results.back());

    // -- Search algorithms --
    std::cout << "\n[Search Latency by Algorithm]\n";
    results.push_back(bench_search("exact", 1000, 32, 10, 500));
    print_result(results.back());
    results.push_back(bench_search("lsh", 1000, 32, 10, 500));
    print_result(results.back());
    results.push_back(bench_search("hnsw", 1000, 32, 10, 500));
    print_result(results.back());

    // -- Larger database --
    std::cout << "\n[Search Latency at Scale]\n";
    results.push_back(bench_search("exact", 5000, 64, 10, 200));
    print_result(results.back());
    results.push_back(bench_search("hnsw", 5000, 64, 10, 200));
    print_result(results.back());

    // -- Update --
    std::cout << "\n[Update Throughput]\n";
    results.push_back(bench_update(1000, 500, 32));
    print_result(results.back());

    // -- Delete --
    std::cout << "\n[Delete Throughput]\n";
    results.push_back(bench_delete(1000, 32));
    print_result(results.back());

    // -- Cache --
    std::cout << "\n[Cache Impact]\n";
    auto [cache_hit, cache_miss] = bench_cache(1000, 32, 1000);
    print_result(cache_hit);
    print_result(cache_miss);
    double speedup = cache_miss.avg_us / cache_hit.avg_us;
    std::cout << "    Cache speedup: " << std::fixed << std::setprecision(1) << speedup << "x\n";

    // -- Concurrency --
    std::cout << "\n[Concurrent Read Throughput]\n";
    results.push_back(bench_concurrent_reads(1000, 32, 1, 200));
    print_result(results.back());
    results.push_back(bench_concurrent_reads(1000, 32, 4, 200));
    print_result(results.back());
    results.push_back(bench_concurrent_reads(1000, 32, 8, 200));
    print_result(results.back());

    // -- Scaling --
    std::cout << "\n[Scaling]\n";
    bench_dimension_scaling(1000, 200);
    bench_size_scaling(32, 200);

    // -- Summary --
    std::cout << "\n========================================\n";
    std::cout << " Performance test complete.\n";
    std::cout << "========================================\n";

    return 0;
}
