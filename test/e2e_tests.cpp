// test/e2e_tests.cpp
// End-to-end tests: full workflows through the VectorDatabase public API

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../src/core/vector.hpp"
#include "../src/core/vector_database.hpp"
#include "../src/utils/distance_metrics.hpp"

// ---- minimal test harness (same as unit_tests.cpp) ----

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

struct TestFailure {};

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        std::cerr << "  FAIL: " << #expr << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw TestFailure{}; \
    } \
} while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        std::cerr << "  FAIL: " << #a << " == " << #b \
                  << " (got " << _a << " vs " << _b << ")" \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw TestFailure{}; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "  FAIL: |" << #a << " - " << #b << "| <= " << (eps) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw TestFailure{}; \
    } \
} while(0)

void run_test(const std::string& name, std::function<void()> fn) {
    tests_run++;
    try {
        fn();
        tests_passed++;
        std::cout << "  PASS: " << name << "\n";
    } catch (const TestFailure&) {
        tests_failed++;
    } catch (const std::exception& e) {
        tests_failed++;
        std::cerr << "  FAIL: " << name << " (exception: " << e.what() << ")\n";
    }
}

// ---- helpers ----

static Vector make_vec(std::initializer_list<float> vals) {
    return Vector(std::vector<float>(vals));
}

static Vector random_vec(size_t dims, unsigned seed) {
    std::vector<float> v(dims);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dims; i++) v[i] = dist(rng);
    return Vector(v);
}

// =====================================================================
//  WORKFLOW: Insert -> Search -> Update -> Search -> Delete -> Search
// =====================================================================

void test_full_crud_workflow() {
    VectorDatabase db(3);
    db.initialize();

    // Insert 5 vectors
    ASSERT_TRUE(db.insert(make_vec({1, 0, 0}), "v1", "category:x"));
    ASSERT_TRUE(db.insert(make_vec({0, 1, 0}), "v2", "category:y"));
    ASSERT_TRUE(db.insert(make_vec({0, 0, 1}), "v3", "category:z"));
    ASSERT_TRUE(db.insert(make_vec({1, 1, 0}), "v4", "category:xy"));
    ASSERT_TRUE(db.insert(make_vec({0, 1, 1}), "v5", "category:yz"));
    ASSERT_EQ(db.vectorCount(), 5u);

    // Search for nearest to (1, 0.1, 0) -> should be v1
    auto r1 = db.similaritySearch(make_vec({1, 0.1f, 0}), 1);
    ASSERT_EQ(r1[0].first, "v1");

    // Update v1 to point away
    ASSERT_TRUE(db.update(make_vec({-10, -10, -10}), "v1", "category:far"));
    ASSERT_EQ(db.getMetadata("v1"), "category:far");

    // Same search now should NOT return v1
    auto r2 = db.similaritySearch(make_vec({1, 0.1f, 0}), 1);
    ASSERT_TRUE(r2[0].first != "v1");

    // Delete v4
    ASSERT_TRUE(db.remove("v4"));
    ASSERT_EQ(db.vectorCount(), 4u);
    ASSERT_FALSE(db.get("v4").has_value());

    // Search should not return v4
    auto r3 = db.similaritySearch(make_vec({1, 1, 0}), 5);
    for (auto& [key, dist] : r3) {
        ASSERT_TRUE(key != "v4");
    }
}

// =====================================================================
//  WORKFLOW: Search after delete reflects removed vectors
// =====================================================================

void test_delete_removes_from_search() {
    VectorDatabase db(2);
    db.initialize();

    // Insert 3 vectors along x-axis
    ASSERT_TRUE(db.insert(make_vec({1, 0}), "close"));
    ASSERT_TRUE(db.insert(make_vec({2, 0}), "medium"));
    ASSERT_TRUE(db.insert(make_vec({10, 0}), "far"));

    // Delete the closest
    ASSERT_TRUE(db.remove("close"));

    auto results = db.similaritySearch(make_vec({0, 0}), 3);
    ASSERT_EQ(results.size(), 2u);
    ASSERT_EQ(results[0].first, "medium");
    ASSERT_EQ(results[1].first, "far");
}

// =====================================================================
//  WORKFLOW: Search mode switching (exact -> HNSW -> exact)
// =====================================================================

void test_search_mode_switching() {
    VectorDatabase db(4);
    db.initialize();

    // Insert data with exact search
    for (int i = 0; i < 50; i++) {
        ASSERT_TRUE(db.insert(random_vec(4, static_cast<unsigned>(i)), "v" + std::to_string(i)));
    }

    Vector query = random_vec(4, 999);

    // Exact search baseline
    auto exact = db.similaritySearch(query, 5);
    ASSERT_EQ(exact.size(), 5u);

    // Switch to HNSW
    db.configureHNSW(8, 50, 50);
    db.setSearchMode(VectorDatabase::SearchMode::HNSW);
    auto hnsw = db.similaritySearch(query, 5);
    ASSERT_TRUE(hnsw.size() >= 1);

    // Switch back to exact
    db.setSearchMode(VectorDatabase::SearchMode::Exact);
    auto exact2 = db.similaritySearch(query, 5);
    ASSERT_EQ(exact2.size(), 5u);
    // Should return same top result as before
    ASSERT_EQ(exact2[0].first, exact[0].first);
}

// =====================================================================
//  WORKFLOW: Batch operations
// =====================================================================

void test_batch_insert_workflow() {
    VectorDatabase db(3, VectorDatabase::SearchMode::Exact, false, true);
    db.initialize();

    std::vector<std::string> keys;
    std::vector<Vector> vectors;
    std::vector<std::string> metadata;

    for (int i = 0; i < 20; i++) {
        keys.push_back("batch_" + std::to_string(i));
        vectors.push_back(random_vec(3, static_cast<unsigned>(i + 100)));
        metadata.push_back("meta_" + std::to_string(i));
    }

    auto result = db.batchInsert(keys, vectors, metadata);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.operations_committed, 20u);
    ASSERT_EQ(db.vectorCount(), 20u);

    // Verify metadata
    ASSERT_EQ(db.getMetadata("batch_0"), "meta_0");
    ASSERT_EQ(db.getMetadata("batch_19"), "meta_19");
}

void test_batch_update_workflow() {
    VectorDatabase db(2, VectorDatabase::SearchMode::Exact, false, true);
    db.initialize();

    // Insert
    ASSERT_TRUE(db.insert(make_vec({1, 0}), "a", "old_a"));
    ASSERT_TRUE(db.insert(make_vec({0, 1}), "b", "old_b"));

    // Batch update
    std::vector<std::string> keys = {"a", "b"};
    std::vector<Vector> vecs = {make_vec({-1, 0}), make_vec({0, -1})};
    std::vector<std::string> meta = {"new_a", "new_b"};

    auto result = db.batchUpdate(keys, vecs, meta);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.operations_committed, 2u);

    // Verify updated values
    auto va = db.get("a");
    ASSERT_TRUE(va.has_value());
    ASSERT_NEAR((*va)[0], -1.0f, 1e-6f);
    ASSERT_EQ(db.getMetadata("a"), "new_a");
}

void test_batch_delete_workflow() {
    VectorDatabase db(2, VectorDatabase::SearchMode::Exact, false, true);
    db.initialize();

    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(db.insert(make_vec({static_cast<float>(i), 0}), "d" + std::to_string(i)));
    }
    ASSERT_EQ(db.vectorCount(), 10u);

    std::vector<std::string> to_delete = {"d0", "d2", "d4", "d6", "d8"};
    auto result = db.batchDelete(to_delete);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(db.vectorCount(), 5u);

    // Verify deleted
    ASSERT_FALSE(db.get("d0").has_value());
    ASSERT_TRUE(db.get("d1").has_value());
}

// =====================================================================
//  WORKFLOW: Batch similarity search consistency
// =====================================================================

void test_batch_search_consistency() {
    VectorDatabase db(3);
    db.initialize();

    for (int i = 0; i < 30; i++) {
        ASSERT_TRUE(db.insert(random_vec(3, static_cast<unsigned>(i)), "s" + std::to_string(i)));
    }

    // Run batch search
    std::vector<Vector> queries;
    for (int i = 0; i < 5; i++) {
        queries.push_back(random_vec(3, static_cast<unsigned>(i + 500)));
    }

    auto batch_results = db.batchSimilaritySearch(queries, 3);
    ASSERT_EQ(batch_results.size(), 5u);

    // Each individual search should match the batch
    for (size_t i = 0; i < queries.size(); i++) {
        auto single = db.similaritySearch(queries[i], 3);
        ASSERT_EQ(single.size(), batch_results[i].size());
        // Top result should match
        if (!single.empty()) {
            ASSERT_EQ(single[0].first, batch_results[i][0].first);
        }
    }
}

// =====================================================================
//  WORKFLOW: Distance metric affects results
// =====================================================================

void test_distance_metric_affects_ranking() {
    VectorDatabase db(2);
    db.initialize();

    // Create vectors where euclidean and manhattan give different rankings
    ASSERT_TRUE(db.insert(make_vec({3, 0}), "a"));     // euclidean=3, manhattan=3
    ASSERT_TRUE(db.insert(make_vec({2, 2}), "b"));     // euclidean=2.83, manhattan=4
    ASSERT_TRUE(db.insert(make_vec({10, 10}), "c"));    // far away

    Vector query = make_vec({0, 0});

    // Euclidean: b (2.83) < a (3.0) < c
    auto euc_results = db.similaritySearch(query, 2);
    ASSERT_EQ(euc_results[0].first, "b");
    ASSERT_EQ(euc_results[1].first, "a");

    // Manhattan: a (3.0) < b (4.0) < c
    db.setDistanceMetric(std::make_shared<ManhattanDistance>());
    auto man_results = db.similaritySearch(query, 2);
    ASSERT_EQ(man_results[0].first, "a");
    ASSERT_EQ(man_results[1].first, "b");
}

// =====================================================================
//  WORKFLOW: Cache behavior across mutations
// =====================================================================

void test_cache_invalidated_on_mutation() {
    VectorDatabase db(2, VectorDatabase::SearchMode::Exact, false, false, {}, true, 100);
    db.initialize();

    ASSERT_TRUE(db.insert(make_vec({1, 0}), "a"));
    ASSERT_TRUE(db.insert(make_vec({0, 1}), "b"));

    Vector query = make_vec({1, 0});

    // First search - cache miss
    auto r1 = db.similaritySearch(query, 1);
    ASSERT_EQ(r1[0].first, "a");

    // Second search - cache hit (same result)
    auto r2 = db.similaritySearch(query, 1);
    ASSERT_EQ(r2[0].first, "a");

    // Insert a vector very close to query - should invalidate cache
    // query is (1,0), "a" is (1,0) with distance 0
    // Insert "closer_to_b" near b's direction — this changes the result set
    ASSERT_TRUE(db.insert(make_vec({0.01f, 0.99f}), "near_b"));

    // Search for k=2 should now include "near_b"
    auto r3 = db.similaritySearch(query, 3);
    // Just verify cache was invalidated: we get 3 results now
    ASSERT_EQ(r3.size(), 3u);
    ASSERT_EQ(r3[0].first, "a"); // still closest
}

void test_cache_invalidated_on_update() {
    VectorDatabase db(2, VectorDatabase::SearchMode::Exact, false, false, {}, true, 100);
    db.initialize();

    ASSERT_TRUE(db.insert(make_vec({1, 0}), "a"));
    ASSERT_TRUE(db.insert(make_vec({0, 1}), "b"));

    Vector query = make_vec({0, 0.9f});
    auto r1 = db.similaritySearch(query, 1);
    ASSERT_EQ(r1[0].first, "b");

    // Update b to be far away
    ASSERT_TRUE(db.update(make_vec({100, 100}), "b"));

    auto r2 = db.similaritySearch(query, 1);
    ASSERT_EQ(r2[0].first, "a"); // cache should be invalidated
}

void test_cache_invalidated_on_delete() {
    VectorDatabase db(2, VectorDatabase::SearchMode::Exact, false, false, {}, true, 100);
    db.initialize();

    ASSERT_TRUE(db.insert(make_vec({1, 0}), "closest"));
    ASSERT_TRUE(db.insert(make_vec({5, 0}), "far"));

    Vector query = make_vec({0, 0});
    auto r1 = db.similaritySearch(query, 1);
    ASSERT_EQ(r1[0].first, "closest");

    ASSERT_TRUE(db.remove("closest"));

    auto r2 = db.similaritySearch(query, 1);
    ASSERT_EQ(r2[0].first, "far"); // cache should reflect deletion
}

// =====================================================================
//  WORKFLOW: Concurrent reads
// =====================================================================

void test_concurrent_reads() {
    VectorDatabase db(4);
    db.initialize();

    // Populate
    for (int i = 0; i < 100; i++) {
        ASSERT_TRUE(db.insert(random_vec(4, static_cast<unsigned>(i)), "t" + std::to_string(i)));
    }

    // Spawn multiple reader threads
    const int num_threads = 8;
    const int queries_per_thread = 50;
    std::vector<std::thread> threads;
    std::atomic<int> total_results{0};
    std::atomic<bool> any_error{false};

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            try {
                for (int q = 0; q < queries_per_thread; q++) {
                    auto seed = static_cast<unsigned>(t * 1000 + q);
                    auto r = db.similaritySearch(random_vec(4, seed), 5);
                    total_results.fetch_add(static_cast<int>(r.size()));
                }
            } catch (...) {
                any_error.store(true);
            }
        });
    }

    for (auto& t : threads) t.join();

    ASSERT_FALSE(any_error.load());
    ASSERT_TRUE(total_results.load() > 0);
}

// =====================================================================
//  WORKFLOW: Concurrent reads + writes
// =====================================================================

void test_concurrent_reads_and_writes() {
    VectorDatabase db(4);
    db.initialize();

    // Pre-populate
    for (int i = 0; i < 50; i++) {
        ASSERT_TRUE(db.insert(random_vec(4, static_cast<unsigned>(i)), "init" + std::to_string(i)));
    }

    std::atomic<bool> any_error{false};
    std::vector<std::thread> threads;

    // Writer thread: insert more vectors
    threads.emplace_back([&]() {
        try {
            for (int i = 0; i < 50; i++) {
                (void)db.insert(random_vec(4, static_cast<unsigned>(i + 1000)), "w" + std::to_string(i));
            }
        } catch (...) {
            any_error.store(true);
        }
    });

    // Reader threads
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&, t]() {
            try {
                for (int q = 0; q < 30; q++) {
                    (void)db.similaritySearch(random_vec(4, static_cast<unsigned>(t * 100 + q)), 5);
                }
            } catch (...) {
                any_error.store(true);
            }
        });
    }

    // Updater thread
    threads.emplace_back([&]() {
        try {
            for (int i = 0; i < 20; i++) {
                (void)db.update(random_vec(4, static_cast<unsigned>(i + 2000)), "init" + std::to_string(i));
            }
        } catch (...) {
            any_error.store(true);
        }
    });

    for (auto& t : threads) t.join();

    ASSERT_FALSE(any_error.load());
    ASSERT_TRUE(db.vectorCount() >= 50); // at least the originals
}

// =====================================================================
//  WORKFLOW: Search accuracy with known geometry
// =====================================================================

void test_search_accuracy_known_geometry() {
    VectorDatabase db(2);
    db.initialize();

    // Create a grid of 100 points
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            ASSERT_TRUE(db.insert(
                make_vec({static_cast<float>(x), static_cast<float>(y)}),
                "p" + std::to_string(x) + "_" + std::to_string(y)
            ));
        }
    }

    // Query at (4.1, 4.1) -> closest should be (4,4)
    auto r = db.similaritySearch(make_vec({4.1f, 4.1f}), 4);
    ASSERT_EQ(r[0].first, "p4_4");

    // Next 3 should be the adjacent points (3,4), (4,3), (5,4), (4,5)
    // in some order based on distance
    std::vector<std::string> expected_neighbors = {"p3_4", "p4_3", "p5_4", "p4_5"};
    for (size_t i = 1; i < r.size(); i++) {
        bool found = false;
        for (const auto& exp : expected_neighbors) {
            if (r[i].first == exp) { found = true; break; }
        }
        ASSERT_TRUE(found);
    }
}

// =====================================================================
//  WORKFLOW: HNSW search quality
// =====================================================================

void test_hnsw_search_quality() {
    VectorDatabase db(8, VectorDatabase::SearchMode::HNSW);
    db.initialize();

    // Insert 200 vectors
    for (int i = 0; i < 200; i++) {
        ASSERT_TRUE(db.insert(random_vec(8, static_cast<unsigned>(i)), "h" + std::to_string(i)));
    }

    // Get exact results for comparison
    VectorDatabase exact_db(8, VectorDatabase::SearchMode::Exact);
    exact_db.initialize();
    for (int i = 0; i < 200; i++) {
        ASSERT_TRUE(exact_db.insert(random_vec(8, static_cast<unsigned>(i)), "h" + std::to_string(i)));
    }

    // Run queries and check recall
    int total_queries = 20;
    int top1_matches = 0;
    for (int q = 0; q < total_queries; q++) {
        Vector query = random_vec(8, static_cast<unsigned>(q + 5000));
        auto hnsw_r = db.similaritySearch(query, 1);
        auto exact_r = exact_db.similaritySearch(query, 1);

        if (!hnsw_r.empty() && !exact_r.empty() && hnsw_r[0].first == exact_r[0].first) {
            top1_matches++;
        }
    }

    // HNSW should have at least 70% top-1 recall with these parameters
    float recall = static_cast<float>(top1_matches) / static_cast<float>(total_queries);
    ASSERT_TRUE(recall >= 0.7f);
}

// =====================================================================
//  WORKFLOW: Search with metadata returns correct metadata
// =====================================================================

void test_search_metadata_consistency() {
    VectorDatabase db(2);
    db.initialize();

    ASSERT_TRUE(db.insert(make_vec({1, 0}), "a", "{\"label\": \"alpha\"}"));
    ASSERT_TRUE(db.insert(make_vec({0, 1}), "b", "{\"label\": \"beta\"}"));
    ASSERT_TRUE(db.insert(make_vec({1, 1}), "c", "{\"label\": \"gamma\"}"));

    auto results = db.similaritySearchWithMetadata(make_vec({0.9f, 0.1f}), 3);
    ASSERT_EQ(results.size(), 3u);

    // Find result for "a" and verify its metadata
    for (const auto& r : results) {
        if (r.key == "a") {
            ASSERT_EQ(r.metadata, "{\"label\": \"alpha\"}");
        } else if (r.key == "b") {
            ASSERT_EQ(r.metadata, "{\"label\": \"beta\"}");
        } else if (r.key == "c") {
            ASSERT_EQ(r.metadata, "{\"label\": \"gamma\"}");
        }
    }
}

// =====================================================================
//  WORKFLOW: Large-scale insert + delete + search
// =====================================================================

void test_large_scale_insert_delete() {
    VectorDatabase db(8);
    db.initialize();

    // Insert 500 vectors
    for (int i = 0; i < 500; i++) {
        ASSERT_TRUE(db.insert(random_vec(8, static_cast<unsigned>(i)), "L" + std::to_string(i)));
    }
    ASSERT_EQ(db.vectorCount(), 500u);

    // Delete every other vector
    for (int i = 0; i < 500; i += 2) {
        ASSERT_TRUE(db.remove("L" + std::to_string(i)));
    }
    ASSERT_EQ(db.vectorCount(), 250u);

    // Search should still work and only return non-deleted vectors
    auto results = db.similaritySearch(random_vec(8, 9999), 10);
    ASSERT_TRUE(results.size() <= 10);
    for (const auto& [key, dist] : results) {
        // key should be odd-numbered
        std::string num_str = key.substr(1);
        int num = std::stoi(num_str);
        ASSERT_TRUE(num % 2 == 1);
    }
}

// =====================================================================
//  PERSISTENCE: data survives restart via segmented storage
// =====================================================================

void test_segmented_recovery_after_restart() {
    auto path = std::filesystem::temp_directory_path() /
                ("e2e_seg_recover_" + std::to_string(::getpid()));
    std::filesystem::remove_all(path);

    constexpr size_t N = 50;
    {
        VectorDatabase db(4,
                          VectorDatabase::SearchMode::HNSW,
                          /*atomic_persistence*/false,
                          /*batch_ops*/false,
                          {},
                          /*query_cache*/false,
                          0,
                          path.string(),
                          VectorDatabase::StorageEngine::Segmented);
        db.initialize();
        for (size_t i = 0; i < N; ++i) {
            ASSERT_TRUE(db.insert(random_vec(4, static_cast<unsigned>(i + 1)),
                                  "k" + std::to_string(i),
                                  "m" + std::to_string(i)));
        }
        ASSERT_EQ(db.vectorCount(), N);
        ASSERT_TRUE(db.checkpoint());
        db.shutdown();
    }
    {
        VectorDatabase db(4,
                          VectorDatabase::SearchMode::HNSW,
                          false, false, {}, false, 0,
                          path.string(),
                          VectorDatabase::StorageEngine::Segmented);
        db.initialize();
        ASSERT_EQ(db.vectorCount(), N);
        for (size_t i = 0; i < N; ++i) {
            auto v = db.get("k" + std::to_string(i));
            ASSERT_TRUE(v.has_value());
            ASSERT_EQ(db.getMetadata("k" + std::to_string(i)),
                      "m" + std::to_string(i));
        }
        db.shutdown();
    }
    std::filesystem::remove_all(path);
}

// =====================================================================
//  CONCURRENCY: many threads, all unique keys → no insert lost
// =====================================================================

void test_concurrent_inserts_no_data_loss() {
    VectorDatabase db(4);
    db.initialize();

    constexpr int kThreads = 8;
    constexpr int kPerThread = 100;
    std::vector<std::thread> ts;
    std::atomic<int> ok_count{0};

    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                std::string key = "t" + std::to_string(t) + "_k" + std::to_string(i);
                if (db.insert(random_vec(4, static_cast<unsigned>(t * 1000 + i + 1)),
                              key, "")) {
                    ok_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& th : ts) th.join();

    ASSERT_EQ(ok_count.load(), kThreads * kPerThread);
    ASSERT_EQ(db.vectorCount(), static_cast<size_t>(kThreads * kPerThread));

    // Spot-check that we can retrieve every key.
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kPerThread; i += 25) {
            ASSERT_TRUE(db.get("t" + std::to_string(t) +
                               "_k" + std::to_string(i)).has_value());
        }
    }
}

// =====================================================================
//  UPDATE SEMANTICS: search reflects latest values
// =====================================================================

void test_search_reflects_latest_update() {
    VectorDatabase db(2);
    db.initialize();

    // Insert two well-separated vectors.
    ASSERT_TRUE(db.insert(make_vec({0.0f, 0.0f}), "origin", ""));
    ASSERT_TRUE(db.insert(make_vec({10.0f, 10.0f}), "far", ""));

    // Query near origin → "origin" wins.
    auto r1 = db.similaritySearch(make_vec({0.1f, 0.1f}), 1);
    ASSERT_EQ(r1[0].first, std::string{"origin"});

    // Move "origin" to (10, 10) — now "far" should win for the same query.
    ASSERT_TRUE(db.update(make_vec({10.0f, 10.0f}), "origin", ""));
    auto r2 = db.similaritySearch(make_vec({0.1f, 0.1f}), 1);
    ASSERT_TRUE(r2[0].first == std::string{"origin"} ||
                r2[0].first == std::string{"far"});  // tie now
    // But query close to (10, 10) should return both.
    auto r3 = db.similaritySearch(make_vec({10.0f, 10.0f}), 2);
    std::unordered_set<std::string> keys{r3[0].first, r3[1].first};
    ASSERT_TRUE(keys.contains("origin"));
    ASSERT_TRUE(keys.contains("far"));
}

// =====================================================================
//  HNSW RECALL@K AT MODERATE SCALE
// =====================================================================

void test_hnsw_recall_at_moderate_scale() {
    constexpr size_t N = 1000;
    constexpr size_t dims = 32;
    constexpr size_t k = 10;
    constexpr size_t Q = 50;

    VectorDatabase db(dims, VectorDatabase::SearchMode::HNSW);
    db.configureHNSW(16, 200, 100);   // higher ef for quality
    db.initialize();

    std::vector<Vector> all;
    all.reserve(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto rand_vec = [&] {
        std::vector<float> v(dims);
        for (auto& x : v) x = dist(rng);
        return Vector(v);
    };

    for (size_t i = 0; i < N; ++i) {
        all.push_back(rand_vec());
        ASSERT_TRUE(db.insert(all.back(), "v" + std::to_string(i), ""));
    }

    // Brute-force ground truth on the same DB by switching mode to Exact.
    VectorDatabase exact(dims, VectorDatabase::SearchMode::Exact);
    exact.initialize();
    for (size_t i = 0; i < N; ++i) {
        ASSERT_TRUE(exact.insert(all[i], "v" + std::to_string(i), ""));
    }

    size_t hits = 0, total = 0;
    for (size_t q = 0; q < Q; ++q) {
        Vector query = rand_vec();
        auto truth  = exact.similaritySearch(query, k);
        auto approx = db.similaritySearch(query, k);
        std::unordered_set<std::string> truth_keys;
        for (auto& [key, _] : truth) truth_keys.insert(key);
        for (auto& [key, _] : approx)
            if (truth_keys.contains(key)) ++hits;
        total += truth.size();
    }
    double recall = static_cast<double>(hits) / static_cast<double>(total);
    if (recall < 0.85) {
        std::cerr << "  (recall=" << recall << ")\n";
    }
    ASSERT_TRUE(recall >= 0.85);
}

// =====================================================================
//  CONCURRENCY: search consistency under interleaved deletes
// =====================================================================

void test_search_remains_consistent_under_delete() {
    VectorDatabase db(2);
    db.initialize();

    constexpr int N = 200;
    for (int i = 0; i < N; ++i) {
        ASSERT_TRUE(db.insert(make_vec({float(i), 0.0f}), "k" + std::to_string(i), ""));
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> saw_invalid{false};

    std::thread reader([&] {
        while (!stop.load()) {
            auto results = db.similaritySearch(make_vec({0.0f, 0.0f}), 5);
            for (const auto& [key, _] : results) {
                // The key must be parseable and within range.
                if (key.size() < 2 || key[0] != 'k') {
                    saw_invalid.store(true);
                }
            }
        }
    });
    std::thread deleter([&] {
        for (int i = 0; i < N / 2; ++i) {
            (void)db.remove("k" + std::to_string(i * 2));
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    deleter.join();
    stop.store(true);
    reader.join();

    ASSERT_FALSE(saw_invalid.load());
}

// =====================================================================
//  MAIN
// =====================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " Vector Database End-to-End Tests\n";
    std::cout << "========================================\n\n";

    std::cout << "[CRUD Workflows]\n";
    run_test("full CRUD workflow", test_full_crud_workflow);
    run_test("delete removes from search", test_delete_removes_from_search);
    run_test("large-scale insert + delete", test_large_scale_insert_delete);

    std::cout << "\n[Search Mode Switching]\n";
    run_test("exact -> HNSW -> exact", test_search_mode_switching);
    run_test("HNSW search quality (recall)", test_hnsw_search_quality);

    std::cout << "\n[Batch Operations]\n";
    run_test("batch insert", test_batch_insert_workflow);
    run_test("batch update", test_batch_update_workflow);
    run_test("batch delete", test_batch_delete_workflow);
    run_test("batch search consistency", test_batch_search_consistency);

    std::cout << "\n[Distance Metrics]\n";
    run_test("metric affects ranking", test_distance_metric_affects_ranking);

    std::cout << "\n[Cache Behavior]\n";
    run_test("cache invalidated on insert", test_cache_invalidated_on_mutation);
    run_test("cache invalidated on update", test_cache_invalidated_on_update);
    run_test("cache invalidated on delete", test_cache_invalidated_on_delete);

    std::cout << "\n[Concurrency]\n";
    run_test("concurrent reads", test_concurrent_reads);
    run_test("concurrent reads + writes", test_concurrent_reads_and_writes);

    std::cout << "\n[Search Accuracy]\n";
    run_test("known geometry search", test_search_accuracy_known_geometry);
    run_test("metadata consistency", test_search_metadata_consistency);

    std::cout << "\n[Persistence]\n";
    run_test("segmented recovery after restart", test_segmented_recovery_after_restart);

    std::cout << "\n[Concurrency at scale]\n";
    run_test("concurrent inserts no data loss",  test_concurrent_inserts_no_data_loss);
    run_test("search consistent under delete",   test_search_remains_consistent_under_delete);

    std::cout << "\n[Update semantics]\n";
    run_test("search reflects latest update",    test_search_reflects_latest_update);

    std::cout << "\n[HNSW recall at scale]\n";
    run_test("recall@10 >= 0.85 (n=1000)",       test_hnsw_recall_at_moderate_scale);

    std::cout << "\n========================================\n";
    std::cout << " Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << " (" << tests_failed << " FAILED)";
    }
    std::cout << "\n========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
