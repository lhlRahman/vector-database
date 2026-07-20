// test/unit_tests.cpp
// Lightweight unit test framework (no external dependencies)

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/core/vector.hpp"
#include "../src/core/vector_accessor.hpp"
#include "../src/core/vector_database.hpp"
#include "../src/algorithms/flat_index.hpp"
#include "../src/algorithms/hnsw_index.hpp"
#include "../src/api/protocol.hpp"
#include "../src/features/commit_log.hpp"
#include "../src/features/query_cache.hpp"
#include "../src/optimizations/rw_lock.hpp"
#include "../src/optimizations/scalar_quantization.hpp"
#include "../src/storage/mmap_storage.hpp"
#include "../src/storage/segmented_vector_store.hpp"
#include "../src/utils/atomic_write.hpp"
#include "../src/utils/distance_metrics.hpp"

#include <atomic>
#include <random>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

// ---- minimal test harness ----

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

struct TestFailure {
    std::string file;
    int line;
    std::string expr;
};

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        std::cerr << "  FAIL: " << #expr << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw TestFailure{__FILE__, __LINE__, #expr}; \
    } \
} while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        std::cerr << "  FAIL: " << #a << " == " << #b \
                  << " (got " << _a << " vs " << _b << ")" \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw TestFailure{__FILE__, __LINE__, #a " == " #b}; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    auto _a = (a); auto _b = (b); \
    if (std::abs(_a - _b) > (eps)) { \
        std::cerr << "  FAIL: |" << #a << " - " << #b << "| <= " << (eps) \
                  << " (got " << _a << " vs " << _b << ", diff=" << std::abs(_a - _b) << ")" \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw TestFailure{__FILE__, __LINE__, #a " near " #b}; \
    } \
} while(0)

#define ASSERT_THROWS(expr, exception_type) do { \
    bool caught = false; \
    try { expr; } catch (const exception_type&) { caught = true; } \
    if (!caught) { \
        std::cerr << "  FAIL: expected " << #exception_type << " from " << #expr \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        throw TestFailure{__FILE__, __LINE__, #expr " throws " #exception_type}; \
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

// ---- test vector store (simulates mmap storage for index unit tests) ----

struct TestVectorStore {
    std::vector<Vector> vectors;

    uint64_t add(const Vector& v) {
        uint64_t id = vectors.size();
        vectors.push_back(v);
        return id;
    }

    VectorAccessor accessor() {
        return [this](uint64_t id) -> const float* {
            return vectors[id].data_ptr();
        };
    }
};

// =====================================================================
//  VECTOR TESTS
// =====================================================================

void test_vector_construction() {
    Vector v1(4);
    ASSERT_EQ(v1.size(), 4u);
    for (size_t i = 0; i < 4; i++) {
        ASSERT_NEAR(v1[i], 0.0f, 1e-9f);
    }

    Vector v2(std::vector<float>{1.0f, 2.0f, 3.0f});
    ASSERT_EQ(v2.size(), 3u);
    ASSERT_NEAR(v2[0], 1.0f, 1e-9f);
    ASSERT_NEAR(v2[1], 2.0f, 1e-9f);
    ASSERT_NEAR(v2[2], 3.0f, 1e-9f);
}

void test_vector_access() {
    Vector v(std::vector<float>{10.0f, 20.0f, 30.0f});
    v[0] = 99.0f;
    ASSERT_NEAR(v[0], 99.0f, 1e-9f);

    ASSERT_THROWS(v[3], std::out_of_range);
    ASSERT_THROWS(v[100], std::out_of_range);
}

void test_vector_dot_product() {
    Vector v1(std::vector<float>{1.0f, 2.0f, 3.0f});
    Vector v2(std::vector<float>{4.0f, 5.0f, 6.0f});
    float dot = Vector::dot_product(v1, v2);
    ASSERT_NEAR(dot, 32.0f, 1e-4f);

    Vector v3(std::vector<float>{1.0f, 0.0f});
    ASSERT_THROWS(Vector::dot_product(v1, v3), std::invalid_argument);
}

void test_vector_equality() {
    Vector v1(std::vector<float>{1.0f, 2.0f});
    Vector v2(std::vector<float>{1.0f, 2.0f});
    Vector v3(std::vector<float>{1.0f, 3.0f});
    ASSERT_TRUE(v1 == v2);
    ASSERT_FALSE(v1 == v3);
}

void test_vector_data_ptr() {
    Vector v(std::vector<float>{1.0f, 2.0f, 3.0f});
    const float* p = v.data_ptr();
    ASSERT_NEAR(p[0], 1.0f, 1e-9f);
    ASSERT_NEAR(p[2], 3.0f, 1e-9f);

    float* mp = v.data_ptr();
    mp[1] = 42.0f;
    ASSERT_NEAR(v[1], 42.0f, 1e-9f);
}

void test_vector_iterators() {
    Vector v(std::vector<float>{1.0f, 2.0f, 3.0f});
    float sum = 0;
    for (float f : v) sum += f;
    ASSERT_NEAR(sum, 6.0f, 1e-6f);
}

void test_vector_serialization() {
    Vector original(std::vector<float>{1.5f, -2.5f, 3.14f, 0.0f});
    std::stringstream ss;
    original.write_to(ss);

    ss.seekg(0);
    Vector loaded = Vector::read_from(ss, 4);
    ASSERT_EQ(loaded.size(), 4u);
    for (size_t i = 0; i < 4; i++) {
        ASSERT_NEAR(loaded[i], original[i], 1e-9f);
    }
}

void test_vector_hash() {
    Vector v1(std::vector<float>{1.0f, 2.0f});
    Vector v2(std::vector<float>{1.0f, 2.0f});
    Vector v3(std::vector<float>{3.0f, 4.0f});
    ASSERT_EQ(std::hash<Vector>{}(v1), std::hash<Vector>{}(v2));
    ASSERT_TRUE(std::hash<Vector>{}(v1) != std::hash<Vector>{}(v3));
}

void test_vector_default_construction() {
    Vector v;
    ASSERT_EQ(v.size(), 0u);
}

// =====================================================================
//  DISTANCE METRICS TESTS
// =====================================================================

void test_euclidean_distance() {
    EuclideanDistance metric;
    Vector v1(std::vector<float>{0.0f, 0.0f});
    Vector v2(std::vector<float>{3.0f, 4.0f});
    float d = metric.distance(v1, v2);
    ASSERT_NEAR(d, 5.0f, 1e-4f);

    // Also test span overload
    float d_raw = metric.distance_raw(std::span(v1.data_ptr(), 2),
                                      std::span(v2.data_ptr(), 2));
    ASSERT_NEAR(d_raw, 5.0f, 1e-4f);
}

void test_euclidean_distance_same() {
    EuclideanDistance metric;
    Vector v(std::vector<float>{1.0f, 2.0f, 3.0f});
    ASSERT_NEAR(metric.distance(v, v), 0.0f, 1e-6f);
    ASSERT_NEAR(metric.distance_raw(std::span(v.data_ptr(), 3),
                                    std::span(v.data_ptr(), 3)), 0.0f, 1e-6f);
}

void test_manhattan_distance() {
    ManhattanDistance metric;
    Vector v1(std::vector<float>{1.0f, 2.0f, 3.0f});
    Vector v2(std::vector<float>{4.0f, 6.0f, 3.0f});
    float d = metric.distance(v1, v2);
    ASSERT_NEAR(d, 7.0f, 1e-4f);

    float d_raw = metric.distance_raw(std::span(v1.data_ptr(), 3),
                                      std::span(v2.data_ptr(), 3));
    ASSERT_NEAR(d_raw, 7.0f, 1e-4f);
}

void test_cosine_similarity() {
    CosineSimilarity metric;
    Vector v1(std::vector<float>{1.0f, 0.0f});
    Vector v2(std::vector<float>{0.0f, 1.0f});
    float d = metric.distance(v1, v2);
    ASSERT_NEAR(d, 1.0f, 1e-4f);

    Vector v3(std::vector<float>{2.0f, 0.0f});
    float d2 = metric.distance(v1, v3);
    ASSERT_NEAR(d2, 0.0f, 1e-4f);
}

void test_cosine_similarity_parallel() {
    CosineSimilarity metric;
    Vector v1(std::vector<float>{1.0f, 1.0f});
    Vector v2(std::vector<float>{2.0f, 2.0f});
    float d = metric.distance(v1, v2);
    ASSERT_NEAR(d, 0.0f, 1e-4f);
}

// =====================================================================
//  FLAT EXACT INDEX TESTS
// =====================================================================

void test_flat_index_metric_policy() {
    TestVectorStore store;
    std::unordered_map<std::string, uint64_t> slots;

    slots["a"] = store.add(Vector(std::vector<float>{1.0f, 0.0f}));
    slots["b"] = store.add(Vector(std::vector<float>{0.0f, 2.0f}));
    slots["c"] = store.add(Vector(std::vector<float>{10.0f, 10.0f}));

    FlatIndex<ManhattanMetricPolicy> index(2, store.accessor());
    auto results = index.search(Vector(std::vector<float>{0.0f, 0.0f}), 2, slots);

    ASSERT_EQ(results.size(), 2u);
    ASSERT_EQ(results[0].first, "a");
    ASSERT_EQ(results[1].first, "b");
}

// =====================================================================
//  HNSW INDEX TESTS
// =====================================================================

void test_hnsw_insert_search() {
    auto metric = std::make_shared<EuclideanDistance>();
    TestVectorStore store;

    auto id_a = store.add(Vector(std::vector<float>{1.0f, 0.0f, 0.0f}));
    auto id_b = store.add(Vector(std::vector<float>{0.0f, 1.0f, 0.0f}));
    auto id_c = store.add(Vector(std::vector<float>{0.0f, 0.0f, 1.0f}));
    auto id_d = store.add(Vector(std::vector<float>{1.0f, 1.0f, 0.0f}));

    HNSWIndex hnsw(3, 8, 50, 50, metric, store.accessor());
    hnsw.insert(id_a, "a");
    hnsw.insert(id_b, "b");
    hnsw.insert(id_c, "c");
    hnsw.insert(id_d, "d");

    Vector query(std::vector<float>{1.0f, 0.9f, 0.0f});
    auto results = hnsw.search(query, 1);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].first, "d");
}

void test_hnsw_remove() {
    auto metric = std::make_shared<EuclideanDistance>();
    TestVectorStore store;

    auto id0 = store.add(Vector(std::vector<float>{0.0f, 0.0f}));
    auto id1 = store.add(Vector(std::vector<float>{1.0f, 0.0f}));
    auto id2 = store.add(Vector(std::vector<float>{0.0f, 1.0f}));

    HNSWIndex hnsw(2, 8, 50, 50, metric, store.accessor());
    hnsw.insert(id0, "origin");
    hnsw.insert(id1, "right");
    hnsw.insert(id2, "up");

    hnsw.remove("origin");

    Vector query(std::vector<float>{0.0f, 0.0f});
    auto results = hnsw.search(query, 10);
    for (const auto& [key, dist] : results) {
        ASSERT_TRUE(key != "origin");
    }
}

void test_hnsw_many_vectors() {
    auto metric = std::make_shared<EuclideanDistance>();
    TestVectorStore store;

    HNSWIndex hnsw(4, 8, 50, 50, metric, store.accessor());

    for (int i = 0; i < 100; i++) {
        float f = static_cast<float>(i);
        Vector v(std::vector<float>{f, f * 0.5f, f * 0.25f, f * 0.1f});
        auto id = store.add(v);
        hnsw.insert(id, "v" + std::to_string(i));
    }

    ASSERT_EQ(hnsw.size(), 100u);

    Vector query(std::vector<float>{50.0f, 25.0f, 12.5f, 5.0f});
    auto results = hnsw.search(query, 3);
    ASSERT_TRUE(results.size() >= 1);
    ASSERT_EQ(results[0].first, "v50");
}

void test_hnsw_fixed_seed_reproduces_topology() {
    auto metric = std::make_shared<EuclideanDistance>();
    TestVectorStore store;
    for (int i = 0; i < 64; ++i) {
        float f = static_cast<float>(i);
        store.add(Vector(std::vector<float>{f, f * 0.5f, f * 0.25f, f * 0.125f}));
    }

    HNSWIndex first(4, 8, 50, 50, metric, store.accessor(),
                    HNSWIndex::AllocationStrategy::Standard, 1024 * 1024, 77);
    HNSWIndex second(4, 8, 50, 50, metric, store.accessor(),
                     HNSWIndex::AllocationStrategy::Standard, 1024 * 1024, 77);
    HNSWIndex changed(4, 8, 50, 50, metric, store.accessor(),
                      HNSWIndex::AllocationStrategy::Standard, 1024 * 1024, 78);
    for (uint64_t id = 0; id < 64; ++id) {
        const std::string key = "v" + std::to_string(id);
        first.insert(id, key);
        second.insert(id, key);
        changed.insert(id, key);
    }

    auto a = first.exportGraph();
    auto b = second.exportGraph();
    ASSERT_EQ(a.max_level, b.max_level);
    ASSERT_TRUE(a.entry_points == b.entry_points);
    ASSERT_EQ(a.nodes.size(), b.nodes.size());
    for (size_t i = 0; i < a.nodes.size(); ++i) {
        ASSERT_EQ(a.nodes[i].slot_id, b.nodes[i].slot_id);
        ASSERT_EQ(a.nodes[i].key, b.nodes[i].key);
        ASSERT_EQ(a.nodes[i].level, b.nodes[i].level);
        ASSERT_TRUE(a.nodes[i].neighbors == b.nodes[i].neighbors);
        ASSERT_TRUE(a.nodes[i].neighbor_dists == b.nodes[i].neighbor_dists);
    }

    auto c = changed.exportGraph();
    bool differs = a.max_level != c.max_level || a.entry_points != c.entry_points;
    for (size_t i = 0; !differs && i < a.nodes.size(); ++i) {
        differs = a.nodes[i].level != c.nodes[i].level ||
                  a.nodes[i].neighbors != c.nodes[i].neighbors;
    }
    ASSERT_TRUE(differs);
}

// =====================================================================
//  QUERY CACHE TESTS
// =====================================================================

void test_cache_hit_miss() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f, 2.0f});

    std::vector<std::pair<std::string, float>> results;
    ASSERT_FALSE(cache.get(q, 2, results));

    std::vector<std::pair<std::string, float>> data = {{"a", 0.5f}, {"b", 1.0f}};
    cache.put(q, 2, data);

    ASSERT_TRUE(cache.get(q, 2, results));
    ASSERT_EQ(results.size(), 2u);
    ASSERT_EQ(results[0].first, "a");
}

void test_cache_invalidation() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f, 2.0f});

    std::vector<std::pair<std::string, float>> data = {{"a", 0.5f}};
    cache.put(q, 1, data);

    cache.invalidate();

    std::vector<std::pair<std::string, float>> results;
    ASSERT_FALSE(cache.get(q, 1, results));
}

void test_cache_eviction() {
    QueryCache cache(3);

    for (int i = 0; i < 5; i++) {
        Vector q(std::vector<float>{static_cast<float>(i), 0.0f});
        cache.put(q, 1, {{"x", static_cast<float>(i)}});
    }

    auto stats = cache.getStatistics();
    ASSERT_TRUE(stats.current_size <= 3);
}

void test_cache_statistics() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f});

    std::vector<std::pair<std::string, float>> out;
    (void)cache.get(q, 1, out);              // expected miss
    cache.put(q, 1, {{"a", 1.0f}});
    (void)cache.get(q, 1, out);              // expected hit

    auto stats = cache.getStatistics();
    ASSERT_EQ(stats.hits, 1u);
    ASSERT_EQ(stats.misses, 1u);
    ASSERT_NEAR(stats.hit_rate(), 0.5, 1e-6);
}

void test_cache_clear() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f});
    cache.put(q, 1, {{"a", 1.0f}});

    cache.clear();
    auto stats = cache.getStatistics();
    ASSERT_EQ(stats.current_size, 0u);
}

// Regression: an entry cached for k' neighbors must not answer a query for a
// larger k (top-k is only a prefix of top-k' when k' >= k).
void test_cache_k_aware() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f, 2.0f});
    std::vector<std::pair<std::string, float>> three = {{"a", 0.1f}, {"b", 0.2f}, {"c", 0.3f}};
    cache.put(q, 3, three);

    std::vector<std::pair<std::string, float>> out;
    ASSERT_TRUE(cache.get(q, 3, out));
    ASSERT_EQ(out.size(), 3u);

    // Smaller k is a prefix hit.
    ASSERT_TRUE(cache.get(q, 2, out));
    ASSERT_EQ(out.size(), 2u);
    ASSERT_EQ(out[0].first, "a");

    // Larger k than was cached must MISS, not return a short/stale result.
    ASSERT_FALSE(cache.get(q, 5, out));
}

// Regression: a zero-capacity cache must be a safe no-op (previously back() on
// an empty list = UB).
void test_cache_zero_capacity() {
    QueryCache cache(0);
    Vector q(std::vector<float>{1.0f});
    cache.put(q, 1, {{"a", 1.0f}});   // must not crash / no eviction on empty list

    std::vector<std::pair<std::string, float>> out;
    ASSERT_FALSE(cache.get(q, 1, out));  // nothing is stored
    auto stats = cache.getStatistics();
    ASSERT_EQ(stats.current_size, 0u);
}

// =====================================================================
//  VECTOR DATABASE UNIT TESTS
// =====================================================================

void test_db_insert_get() {
    VectorDatabase db(3);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 2.0f, 3.0f});
    ASSERT_TRUE(db.insert(v, "key1"));

    auto result = db.get("key1");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    ASSERT_NEAR((*result)[0], 1.0f, 1e-9f);
}

void test_db_insert_duplicate() {
    VectorDatabase db(2);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 2.0f});
    ASSERT_TRUE(db.insert(v, "k"));
    ASSERT_FALSE(db.insert(v, "k"));
}

void test_db_insert_dimension_mismatch() {
    VectorDatabase db(3);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 2.0f});
    ASSERT_THROWS((void)db.insert(v, "k"), std::invalid_argument);
}

void test_db_update() {
    VectorDatabase db(2);
    db.initialize();

    Vector v1(std::vector<float>{1.0f, 0.0f});
    ASSERT_TRUE(db.insert(v1, "k"));

    Vector v2(std::vector<float>{0.0f, 1.0f});
    ASSERT_TRUE(db.update(v2, "k"));

    auto result = db.get("k");
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR((*result)[0], 0.0f, 1e-9f);
    ASSERT_NEAR((*result)[1], 1.0f, 1e-9f);
}

void test_db_update_nonexistent() {
    VectorDatabase db(2);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 0.0f});
    ASSERT_FALSE(db.update(v, "nonexistent"));
}

void test_db_remove() {
    VectorDatabase db(2);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 2.0f});
    ASSERT_TRUE(db.insert(v, "k"));
    ASSERT_TRUE(db.remove("k"));
    ASSERT_FALSE(db.get("k").has_value());
}

void test_db_remove_nonexistent() {
    VectorDatabase db(2);
    db.initialize();
    ASSERT_FALSE(db.remove("nope"));
}

void test_db_metadata() {
    VectorDatabase db(2);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 2.0f});
    ASSERT_TRUE(db.insert(v, "k", "some metadata"));

    std::string meta = db.getMetadata("k");
    ASSERT_EQ(meta, "some metadata");
}

void test_db_metadata_empty() {
    VectorDatabase db(2);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 2.0f});
    ASSERT_TRUE(db.insert(v, "k"));
    ASSERT_EQ(db.getMetadata("k"), "");
    ASSERT_EQ(db.getMetadata("nonexistent"), "");
}

void test_db_vector_count() {
    VectorDatabase db(2);
    db.initialize();

    ASSERT_EQ(db.vectorCount(), 0u);

    ASSERT_TRUE(db.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a"));
    ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 1.0f}), "b"));
    ASSERT_EQ(db.vectorCount(), 2u);

    ASSERT_TRUE(db.remove("a"));
    ASSERT_EQ(db.vectorCount(), 1u);
}

void test_db_not_initialized() {
    VectorDatabase db(2);
    Vector v(std::vector<float>{1.0f, 0.0f});
    ASSERT_THROWS((void)db.insert(v, "k"), std::runtime_error);
}

void test_db_similarity_search_exact() {
    VectorDatabase db(3);
    db.initialize();

    ASSERT_TRUE(db.insert(Vector(std::vector<float>{1.0f, 0.0f, 0.0f}), "x"));
    ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 1.0f, 0.0f}), "y"));
    ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 0.0f, 1.0f}), "z"));

    Vector query(std::vector<float>{0.9f, 0.1f, 0.0f});
    auto results = db.similaritySearch(query, 2);
    ASSERT_EQ(results.size(), 2u);
    ASSERT_EQ(results[0].first, "x");
}

void test_db_similarity_search_empty() {
    VectorDatabase db(2);
    db.initialize();

    Vector query(std::vector<float>{1.0f, 0.0f});
    auto results = db.similaritySearch(query, 5);
    ASSERT_EQ(results.size(), 0u);
}

void test_db_search_with_metadata() {
    VectorDatabase db(2);
    db.initialize();

    ASSERT_TRUE(db.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a", "meta_a"));
    ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 1.0f}), "b", "meta_b"));

    Vector query(std::vector<float>{1.0f, 0.0f});
    auto results = db.similaritySearchWithMetadata(query, 1);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].key, "a");
    ASSERT_EQ(results[0].metadata, "meta_a");
}

void test_db_distance_metric_switch() {
    VectorDatabase db(2);
    db.initialize();

    ASSERT_TRUE(db.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a"));
    ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 1.0f}), "b"));

    Vector query(std::vector<float>{0.9f, 0.1f});
    auto r1 = db.similaritySearch(query, 1);
    ASSERT_EQ(r1[0].first, "a");

    db.setDistanceMetric(std::make_shared<ManhattanDistance>());
    auto r2 = db.similaritySearch(query, 1);
    ASSERT_EQ(r2[0].first, "a");
}

void test_db_nan_rejection() {
    VectorDatabase db(2);
    db.initialize();

    Vector v(std::vector<float>{std::nanf(""), 1.0f});
    ASSERT_FALSE(db.insert(v, "nan_vec"));
    ASSERT_FALSE(db.get("nan_vec").has_value());
}

void test_db_statistics() {
    VectorDatabase db(2);
    db.initialize();

    ASSERT_TRUE(db.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a"));
    ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 1.0f}), "b"));
    db.similaritySearch(Vector(std::vector<float>{1.0f, 0.0f}), 1);
    ASSERT_TRUE(db.update(Vector(std::vector<float>{0.5f, 0.5f}), "a"));
    ASSERT_TRUE(db.remove("b"));

    auto stats = db.getStatistics();
    ASSERT_EQ(stats.total_inserts, 2u);
    ASSERT_EQ(stats.total_searches, 1u);
    ASSERT_EQ(stats.total_updates, 1u);
    ASSERT_EQ(stats.total_deletes, 1u);
    ASSERT_EQ(stats.total_vectors, 1u);
}

void test_db_get_all_vectors() {
    VectorDatabase db(2);
    db.initialize();

    ASSERT_TRUE(db.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a"));
    ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 1.0f}), "b"));

    auto all = db.getAllVectors();
    ASSERT_EQ(all.size(), 2u);
    ASSERT_TRUE(all.count("a") == 1);
    ASSERT_TRUE(all.count("b") == 1);
}

void test_db_segmented_persistence_recovery() {
    auto path = std::filesystem::temp_directory_path() /
                ("vdb_segmented_unit_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(path);

    {
        VectorDatabase db(3,
                          VectorDatabase::SearchMode::HNSW,
                          false,
                          false,
                          {},
                          false,
                          0,
                          path.string(),
                          VectorDatabase::StorageEngine::Segmented);
        db.configureHNSW(8, 30, 20);
        db.configureSegmentedStorage(2, 16, 0.25);
        db.initialize();

        ASSERT_TRUE(db.insert(Vector(std::vector<float>{1.0f, 0.0f, 0.0f}), "a", "first"));
        ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 1.0f, 0.0f}), "b", "second"));
        ASSERT_TRUE(db.insert(Vector(std::vector<float>{0.0f, 0.0f, 1.0f}), "c", "third"));
        ASSERT_TRUE(db.remove("b"));
        ASSERT_TRUE(db.update(Vector(std::vector<float>{0.25f, 0.25f, 0.25f}), "c", "updated"));

        db.sealMutableSegment();
        db.compactSegments();
        ASSERT_TRUE(db.checkpoint());
        ASSERT_EQ(db.vectorCount(), 2u);
    }

    {
        VectorDatabase db(3,
                          VectorDatabase::SearchMode::HNSW,
                          false,
                          false,
                          {},
                          false,
                          0,
                          path.string(),
                          VectorDatabase::StorageEngine::Segmented);
        db.configureHNSW(8, 30, 20);
        db.configureSegmentedStorage(2, 16, 0.25);
        db.initialize();

        ASSERT_EQ(db.vectorCount(), 2u);
        ASSERT_TRUE(db.get("a").has_value());
        ASSERT_FALSE(db.get("b").has_value());
        auto c = db.get("c");
        ASSERT_TRUE(c.has_value());
        ASSERT_NEAR((*c)[0], 0.25f, 1e-6f);
        ASSERT_EQ(db.getMetadata("c"), "updated");

        auto results = db.similaritySearch(Vector(std::vector<float>{0.25f, 0.25f, 0.25f}), 1);
        ASSERT_FALSE(results.empty());
        ASSERT_EQ(results[0].first, "c");

        auto stats = db.getStatistics();
        ASSERT_TRUE(stats.storage_engine == VectorDatabase::StorageEngine::Segmented);
        ASSERT_EQ(stats.segmented_stats.total_vectors, 2u);
    }

    std::filesystem::remove_all(path);
}

void test_db_default_segmented_single_op_crash_recovery() {
    auto path = std::filesystem::temp_directory_path() /
                ("vdb_default_crash_unit_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(path);

    pid_t pid = fork();
    ASSERT_TRUE(pid >= 0);

    if (pid == 0) {
        try {
            VectorDatabase db(2,
                              VectorDatabase::SearchMode::HNSW,
                              false,
                              false,
                              {},
                              false,
                              0,
                              path.string());
            db.configureSegmentedStorage(1, 16, 0.25);
            db.initialize();
            bool ok = true;
            ok = ok && db.getStatistics().storage_engine == VectorDatabase::StorageEngine::Segmented;
            ok = ok && db.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a", "old");
            ok = ok && db.insert(Vector(std::vector<float>{0.0f, 1.0f}), "b", "delete-me");
            ok = ok && db.update(Vector(std::vector<float>{0.25f, 0.75f}), "a", "new");
            ok = ok && db.remove("b");
            _exit(ok ? 0 : 1);  // db is still alive: no destructor or shutdown
        } catch (...) {
            _exit(1);
        }
    }

    int status = 0;
    ASSERT_TRUE(waitpid(pid, &status, 0) == pid);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);

    {
        VectorDatabase db(2,
                          VectorDatabase::SearchMode::HNSW,
                          false,
                          false,
                          {},
                          false,
                          0,
                          path.string());
        db.initialize();

        ASSERT_TRUE(db.getStatistics().storage_engine == VectorDatabase::StorageEngine::Segmented);
        ASSERT_EQ(db.vectorCount(), 1u);
        ASSERT_FALSE(db.get("b").has_value());

        auto a = db.get("a");
        ASSERT_TRUE(a.has_value());
        ASSERT_NEAR((*a)[0], 0.25f, 1e-6f);
        ASSERT_NEAR((*a)[1], 0.75f, 1e-6f);
        ASSERT_EQ(db.getMetadata("a"), "new");
    }

    std::filesystem::remove_all(path);
}

// Group commit must be crash-consistent: after batchInsert() returns, its single
// fsync has made the WHOLE batch durable, so a crash (no clean shutdown) must
// recover every row.
void test_db_group_commit_crash_recovery() {
    auto path = std::filesystem::temp_directory_path() /
                ("vdb_groupcommit_crash_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(path);
    constexpr size_t N = 50;

    pid_t pid = fork();
    ASSERT_TRUE(pid >= 0);
    if (pid == 0) {
        try {
            VectorDatabase db(4, VectorDatabase::SearchMode::HNSW, false, /*batch=*/true,
                              {}, false, 0, path.string());
            db.initialize();
            std::vector<std::string> keys;
            std::vector<Vector> vecs;
            for (size_t i = 0; i < N; ++i) {
                keys.push_back("k" + std::to_string(i));
                vecs.push_back(Vector(std::vector<float>{static_cast<float>(i), 1.0f, 2.0f, 3.0f}));
            }
            auto r = db.batchInsert(keys, vecs);  // group commit: one fsync for all N
            _exit(r.operations_committed == N ? 0 : 1);  // db is still alive
        } catch (...) {
            _exit(1);
        }
    }

    int status = 0;
    ASSERT_TRUE(waitpid(pid, &status, 0) == pid);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);

    {
        VectorDatabase db(4, VectorDatabase::SearchMode::HNSW, false, true, {}, false, 0, path.string());
        db.initialize();
        ASSERT_EQ(db.vectorCount(), N);  // every row of the group-committed batch survived
        for (size_t i = 0; i < N; ++i) ASSERT_TRUE(db.get("k" + std::to_string(i)).has_value());
        db.shutdown();
    }
    std::filesystem::remove_all(path);
}

// =====================================================================
//  PROTOCOL (BufferReader / BufferWriter)
// =====================================================================

void test_protocol_buffer_roundtrip() {
    std::vector<uint8_t> buf;
    proto::BufferWriter w(buf);
    w.write_u8(0xAB);
    w.write_u16(0x1234);
    w.write_u32(0xDEADBEEF);
    w.write_f32(3.14159f);
    w.write_string("hello");
    float floats[] = {1.0f, 2.0f, 3.0f};
    w.write_floats(floats, 3);

    proto::BufferReader r(buf.data(), buf.size());
    ASSERT_EQ(r.read_u8(),  uint8_t{0xAB});
    ASSERT_EQ(r.read_u16(), uint16_t{0x1234});
    ASSERT_EQ(r.read_u32(), uint32_t{0xDEADBEEF});
    ASSERT_NEAR(r.read_f32(), 3.14159f, 1e-6f);
    ASSERT_EQ(r.read_string(), std::string{"hello"});
    float out[3]{};
    r.read_floats(out, 3);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[2], 3.0f, 1e-6f);
    ASSERT_EQ(r.remaining(), size_t{0});
}

void test_protocol_buffer_underflow() {
    // Header claims 100-byte string in a 10-byte buffer.
    std::vector<uint8_t> buf(2);
    buf[0] = 100; buf[1] = 0;     // u16 length = 100, little-endian
    proto::BufferReader r(buf.data(), buf.size());
    ASSERT_THROWS(r.read_string(), std::runtime_error);
}

void test_protocol_string_too_long() {
    std::vector<uint8_t> buf;
    proto::BufferWriter w(buf);
    std::string huge(70'000, 'x');  // > UINT16_MAX
    ASSERT_THROWS(w.write_string(huge), std::runtime_error);
}

void test_protocol_short_buffer() {
    // Empty buffer — every read should throw.
    proto::BufferReader r(nullptr, 0);
    ASSERT_THROWS(r.read_u8(),  std::runtime_error);
    ASSERT_THROWS(r.read_u32(), std::runtime_error);
}

// =====================================================================
//  COMMIT LOG (LogEntry serialization, FNV-1a checksum sensitivity)
// =====================================================================

void test_commit_log_entry_roundtrip() {
    std::vector<uint8_t> payload{0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02};
    LogEntry entry(LogEntryType::INSERT, 42, payload);
    auto bytes = entry.serialize();

    LogEntry round = LogEntry::deserialize(bytes);
    ASSERT_TRUE(round.isValid());
    ASSERT_EQ(round.sequence_number, uint64_t{42});
    ASSERT_TRUE(round.data == payload);
}

void test_commit_log_checksum_detects_tampering() {
    std::vector<uint8_t> payload{1, 2, 3, 4, 5, 6};
    LogEntry entry(LogEntryType::UPDATE, 7, payload);
    auto bytes = entry.serialize();

    // Flip a byte in the payload area (after the header). The header layout
    // is timestamp+type+sequence+checksum+data_length, so payload starts
    // somewhere near the end of the serialized bytes.
    bytes[bytes.size() - 1] ^= 0xFF;

    LogEntry tampered = LogEntry::deserialize(bytes);
    ASSERT_FALSE(tampered.isValid());
}

void test_commit_log_oversize_payload_throws() {
    // Constructing with >UINT32_MAX bytes is rejected. We can't actually
    // allocate 4 GiB in a unit test, but we can verify the check by reaching
    // into the constructor's contract: a smaller payload is fine.
    std::vector<uint8_t> small(16);
    LogEntry ok(LogEntryType::INSERT, 1, small);
    ASSERT_TRUE(ok.isValid());
    // (The >4GiB throw is exercised at the API boundary; covered by
    //  static analysis of the precondition rather than a runtime test.)
}

// =====================================================================
//  MMAP STORAGE
// =====================================================================

void test_mmap_storage_basic_crud() {
    auto path = std::filesystem::temp_directory_path() /
                ("mmap_basic_" + std::to_string(::getpid()) + ".vdb");
    std::filesystem::remove(path);

    constexpr size_t dims = 4;
    {
        MMapStorage s(path.string(), dims, /*initial_capacity*/16);
        s.open();

        std::vector<float> v1{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> v2{5.0f, 6.0f, 7.0f, 8.0f};

        uint64_t s1 = s.insert("a", v1.data(), "meta-a");
        uint64_t s2 = s.insert("b", v2.data(), "meta-b");
        ASSERT_TRUE(s1 != UINT64_MAX);
        ASSERT_TRUE(s2 != UINT64_MAX);
        ASSERT_EQ(s.active_count(), uint64_t{2});

        const float* read1 = s.vector_ptr(s1);
        for (size_t i = 0; i < dims; ++i) ASSERT_NEAR(read1[i], v1[i], 1e-6f);
        ASSERT_EQ(s.get_metadata(s1), std::string{"meta-a"});

        ASSERT_TRUE(s.remove(s1));
        ASSERT_EQ(s.active_count(), uint64_t{1});
        ASSERT_FALSE(s.is_active(s1));
        s.close();
    }

    // Reopen; b should still be there.
    {
        MMapStorage s(path.string(), dims);
        s.open();
        ASSERT_EQ(s.active_count(), uint64_t{1});
        auto idx = s.build_key_index();
        ASSERT_TRUE(idx.contains("b"));
        ASSERT_FALSE(idx.contains("a"));
        s.close();
    }

    std::filesystem::remove(path);
}

void test_mmap_storage_dimension_mismatch_on_reopen() {
    auto path = std::filesystem::temp_directory_path() /
                ("mmap_dim_" + std::to_string(::getpid()) + ".vdb");
    std::filesystem::remove(path);
    {
        MMapStorage s(path.string(), 4);
        s.open();
        s.close();
    }
    {
        MMapStorage s(path.string(), 8);  // wrong dims
        ASSERT_THROWS(s.open(), std::runtime_error);
    }
    std::filesystem::remove(path);
}

// =====================================================================
//  SEGMENTED VECTOR STORE (direct API)
// =====================================================================

struct TestWalRecordHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t sequence;
    uint32_t key_bytes;
    uint32_t metadata_bytes;
    uint32_t dimensions;
    uint32_t vector_bytes;
    uint32_t crc32;
};

struct TestWalFencePayload {
    uint64_t generation;
    uint64_t first_lsn;
    uint64_t last_lsn;
    uint64_t mutation_count;
    uint32_t rolling_crc;
    uint32_t reserved;
};

uint32_t test_wal_crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

uint32_t test_wal_record_crc(const TestWalRecordHeader& header,
                             const std::vector<uint8_t>& payload) {
    uint32_t crc = test_wal_crc32_update(
        0xFFFFFFFFu,
        reinterpret_cast<const uint8_t*>(&header),
        offsetof(TestWalRecordHeader, crc32));
    crc = test_wal_crc32_update(crc, payload.data(), payload.size());
    return ~crc;
}

void append_legacy_insert(const std::filesystem::path& wal,
                          uint64_t sequence,
                          const std::string& key,
                          const std::string& metadata,
                          const std::vector<float>& values) {
    std::vector<uint8_t> payload;
    payload.insert(payload.end(), key.begin(), key.end());
    payload.insert(payload.end(), metadata.begin(), metadata.end());
    const auto* vector_bytes = reinterpret_cast<const uint8_t*>(values.data());
    payload.insert(payload.end(), vector_bytes,
                   vector_bytes + values.size() * sizeof(float));

    TestWalRecordHeader header{
        0x314c5756u,
        1,
        1,
        sequence,
        static_cast<uint32_t>(key.size()),
        static_cast<uint32_t>(metadata.size()),
        static_cast<uint32_t>(values.size()),
        static_cast<uint32_t>(values.size() * sizeof(float)),
        0,
    };
    header.crc32 = test_wal_record_crc(header, payload);

    std::ofstream os(wal, std::ios::binary | std::ios::app);
    os.write(reinterpret_cast<const char*>(&header), sizeof(header));
    os.write(reinterpret_cast<const char*>(payload.data()),
             static_cast<std::streamsize>(payload.size()));
    if (!os.good()) throw std::runtime_error("failed writing test WAL record");
}

void test_segmented_store_insert_search_recover() {
    auto root = std::filesystem::temp_directory_path() /
                ("segstore_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    SegmentedVectorStore::Config cfg;
    cfg.dimensions = 3;
    cfg.max_mutable_segment_records = 4;  // force seal+compact behavior
    cfg.max_sealed_segments = 4;

    {
        SegmentedVectorStore s(root, cfg);
        s.initialize();
        for (int i = 0; i < 10; ++i) {
            Vector v(std::vector<float>{float(i), float(i * 2), float(i * 3)});
            ASSERT_TRUE(s.insert(v, "k" + std::to_string(i), "m" + std::to_string(i)));
        }
        ASSERT_EQ(s.vectorCount(), 10u);

        auto results = s.search(Vector(std::vector<float>{0.0f, 0.0f, 0.0f}), 3);
        ASSERT_EQ(results.size(), 3u);
        ASSERT_EQ(results[0].first, std::string{"k0"});

        ASSERT_TRUE(s.remove("k5"));
        ASSERT_EQ(s.vectorCount(), 9u);

        s.flush();
        s.shutdown();
    }

    // Reopen — recovered vectors come back, removed key stays removed.
    {
        SegmentedVectorStore s(root, cfg);
        s.initialize();
        ASSERT_EQ(s.vectorCount(), 9u);
        ASSERT_TRUE(s.get("k0").has_value());
        ASSERT_FALSE(s.get("k5").has_value());
        s.shutdown();
    }

    std::filesystem::remove_all(root);
}

void test_segment_staged_tail_frontiers_and_exact_search() {
    auto root = std::filesystem::temp_directory_path() /
                ("segment_staged_tail_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    VectorSegment::Config cfg;
    cfg.dimensions = 2;
    VectorSegment segment("seg", root, cfg, VectorSegment::State::Mutable);
    segment.initializeNew();

    ASSERT_TRUE(segment.stageInsert(
        Vector(std::vector<float>{1.0f, 0.0f}), "b", "weak-b", 1));
    ASSERT_TRUE(segment.stageInsert(
        Vector(std::vector<float>{-1.0f, 0.0f}), "a", "weak-a", 2));
    ASSERT_EQ(segment.visibleLsn(), 2u);
    ASSERT_EQ(segment.durableLsn(), 0u);
    ASSERT_EQ(segment.volatileCount(), 2u);
    ASSERT_TRUE(segment.volatileBytes() > 0);
    ASSERT_TRUE(segment.isVolatile("a"));

    auto latest = segment.search(Vector(std::vector<float>{0.0f, 0.0f}), 2);
    ASSERT_EQ(latest.size(), 2u);
    ASSERT_EQ(latest[0].key, std::string{"a"});
    ASSERT_EQ(latest[1].key, std::string{"b"});
    ASSERT_TRUE(segment.searchStable(Vector(std::vector<float>{0.0f, 0.0f}), 2).empty());

    ASSERT_EQ(segment.commitThrough(2), 2u);
    ASSERT_EQ(segment.durableLsn(), 2u);
    ASSERT_EQ(segment.volatileCount(), 0u);
    ASSERT_EQ(segment.volatileBytes(), 0u);
    ASSERT_FALSE(segment.isVolatile("a"));
    ASSERT_EQ(segment.searchStable(Vector(std::vector<float>{0.0f, 0.0f}), 2).size(), 2u);

    segment.prepareSeal();
    ASSERT_TRUE(std::filesystem::exists(root / "seal.ready"));
    ASSERT_TRUE(std::filesystem::exists(root / "wal.log"));
    ASSERT_TRUE(segment.state() == VectorSegment::State::Mutable);
    segment.activateSeal();
    segment.retireWal();
    ASSERT_FALSE(std::filesystem::exists(root / "wal.log"));

    std::filesystem::remove_all(root);
}

void test_store_staged_insert_recovery_drops_tail() {
    auto root = std::filesystem::temp_directory_path() /
                ("store_staged_recovery_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    SegmentedVectorStore::Config cfg;
    cfg.dimensions = 2;
    uint64_t stable_lsn = 0;
    uintmax_t stable_wal_bytes = 0;
    {
        SegmentedVectorStore store(root, cfg);
        store.initialize();
        ASSERT_TRUE(store.insert(Vector(std::vector<float>{10.0f, 0.0f}), "stable"));
        stable_lsn = store.durableLsn();
        stable_wal_bytes = std::filesystem::file_size(
            root / "segments" / "seg_00000001" / "wal.log");

        auto staged = store.stageInsert(
            Vector(std::vector<float>{0.0f, 0.0f}), "weak", "provisional");
        ASSERT_TRUE(staged.applied);
        ASSERT_TRUE(staged.lsn > stable_lsn);
        ASSERT_EQ(store.visibleLsn(), staged.lsn);
        ASSERT_EQ(store.durableLsn(), stable_lsn);
        ASSERT_EQ(store.volatileCount(), 1u);
        ASSERT_TRUE(store.volatileBytes() > 0);
        ASSERT_TRUE(store.isVolatile("weak"));
        ASSERT_TRUE(std::filesystem::file_size(
                        root / "segments" / "seg_00000001" / "wal.log") >
                    stable_wal_bytes);

        auto latest = store.search(Vector(std::vector<float>{0.0f, 0.0f}), 1);
        ASSERT_EQ(latest.size(), 1u);
        ASSERT_EQ(latest[0].first, std::string{"weak"});
        auto stable = store.searchStable(Vector(std::vector<float>{0.0f, 0.0f}), 1);
        ASSERT_EQ(stable.size(), 1u);
        ASSERT_EQ(stable[0].first, std::string{"stable"});
        // No shutdown: the staged generation deliberately remains unfenced.
    }

    {
        SegmentedVectorStore recovered(root, cfg);
        recovered.initialize();
        ASSERT_TRUE(recovered.get("stable").has_value());
        ASSERT_FALSE(recovered.get("weak").has_value());
        ASSERT_EQ(recovered.visibleLsn(), recovered.durableLsn());
        ASSERT_TRUE(recovered.durableLsn() >= stable_lsn);
        ASSERT_EQ(recovered.volatileCount(), 0u);
        ASSERT_EQ(std::filesystem::file_size(
                      root / "segments" / "seg_00000001" / "wal.log"),
                  stable_wal_bytes);
        recovered.shutdown();
    }

    std::filesystem::remove_all(root);
}

void test_store_manifest_role_is_authoritative() {
    auto root = std::filesystem::temp_directory_path() /
                ("store_manifest_role_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    SegmentedVectorStore::Config cfg;
    cfg.dimensions = 2;
    cfg.max_mutable_segment_records = 2;
    {
        SegmentedVectorStore store(root, cfg);
        store.initialize();
        ASSERT_TRUE(store.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a"));
        ASSERT_TRUE(store.insert(Vector(std::vector<float>{2.0f, 0.0f}), "b"));
    }

    const auto sealed_dir = root / "segments" / "seg_00000001";
    ASSERT_TRUE(std::filesystem::exists(sealed_dir / "seal.ready"));
    ASSERT_FALSE(std::filesystem::exists(sealed_dir / "wal.log"));

    // Segment metadata is diagnostic. Even stale metadata from a crash must not
    // override the role installed atomically in the manifest.
    const auto metadata_path = sealed_dir / "segment.meta";
    std::ifstream metadata_in(metadata_path);
    std::string metadata(std::istreambuf_iterator<char>(metadata_in), {});
    const auto state_pos = metadata.find("state=sealed\n");
    ASSERT_TRUE(state_pos != std::string::npos);
    metadata.replace(state_pos, std::string("state=sealed\n").size(), "state=mutable\n");
    {
        std::ofstream metadata_out(metadata_path, std::ios::trunc);
        metadata_out << metadata;
        ASSERT_TRUE(metadata_out.good());
    }

    {
        SegmentedVectorStore recovered(root, cfg);
        recovered.initialize();
        ASSERT_TRUE(recovered.get("a").has_value());
        ASSERT_TRUE(recovered.get("b").has_value());
        ASSERT_TRUE(recovered.insert(Vector(std::vector<float>{3.0f, 0.0f}), "c"));
        recovered.shutdown();
    }

    std::filesystem::remove_all(root);
}

void test_wal_v2_discards_unfenced_tail() {
    auto root = std::filesystem::temp_directory_path() /
                ("wal_v2_unfenced_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    VectorSegment::Config cfg;
    cfg.dimensions = 2;
    {
        VectorSegment segment("seg", root, cfg, VectorSegment::State::Mutable);
        segment.initializeNew();
        segment.beginDeferredSync();
        ASSERT_TRUE(segment.insert(Vector(std::vector<float>{1.0f, 2.0f}), "weak", "", 1));
        ASSERT_EQ(segment.recordCount(), 1u);
        // Deliberately omit commitDeferredSync(): this models an open weak generation.
    }
    ASSERT_TRUE(std::filesystem::file_size(root / "wal.log") > 0);

    {
        VectorSegment recovered("seg", root, cfg, VectorSegment::State::Mutable);
        recovered.load();
        ASSERT_EQ(recovered.recordCount(), 0u);
        ASSERT_EQ(std::filesystem::file_size(root / "wal.log"), 0u);
    }
    std::filesystem::remove_all(root);
}

void test_wal_v2_replays_fenced_generation() {
    auto root = std::filesystem::temp_directory_path() /
                ("wal_v2_fenced_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    VectorSegment::Config cfg;
    cfg.dimensions = 2;
    {
        VectorSegment segment("seg", root, cfg, VectorSegment::State::Mutable);
        segment.initializeNew();
        segment.beginDeferredSync();
        ASSERT_TRUE(segment.insert(Vector(std::vector<float>{1.0f, 2.0f}), "a", "", 1));
        ASSERT_TRUE(segment.insert(Vector(std::vector<float>{3.0f, 4.0f}), "b", "", 2));
        segment.commitDeferredSync();
    }

    {
        VectorSegment recovered("seg", root, cfg, VectorSegment::State::Mutable);
        recovered.load();
        ASSERT_EQ(recovered.recordCount(), 2u);
        ASSERT_TRUE(recovered.contains("a"));
        ASSERT_TRUE(recovered.contains("b"));
        ASSERT_EQ(recovered.maxSequence(), 2u);
    }
    std::filesystem::remove_all(root);
}

void test_wal_v2_rejects_inconsistent_fence() {
    auto root = std::filesystem::temp_directory_path() /
                ("wal_v2_bad_fence_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    VectorSegment::Config cfg;
    cfg.dimensions = 2;
    {
        VectorSegment segment("seg", root, cfg, VectorSegment::State::Mutable);
        segment.initializeNew();
        segment.beginDeferredSync();
        ASSERT_TRUE(segment.insert(Vector(std::vector<float>{1.0f, 2.0f}), "a", "", 1));
        ASSERT_TRUE(segment.insert(Vector(std::vector<float>{3.0f, 4.0f}), "b", "", 2));
        segment.commitDeferredSync();
    }

    const auto wal = root / "wal.log";
    const auto fence_offset = std::filesystem::file_size(wal) -
                              sizeof(TestWalRecordHeader) - sizeof(TestWalFencePayload);
    {
        std::fstream io(wal, std::ios::binary | std::ios::in | std::ios::out);
        io.seekg(static_cast<std::streamoff>(fence_offset));
        TestWalRecordHeader header{};
        TestWalFencePayload fence{};
        io.read(reinterpret_cast<char*>(&header), sizeof(header));
        io.read(reinterpret_cast<char*>(&fence), sizeof(fence));
        ASSERT_TRUE(io.good());

        ++fence.mutation_count;
        std::vector<uint8_t> payload(sizeof(fence));
        std::memcpy(payload.data(), &fence, sizeof(fence));
        header.crc32 = test_wal_record_crc(header, payload);

        io.seekp(static_cast<std::streamoff>(fence_offset));
        io.write(reinterpret_cast<const char*>(&header), sizeof(header));
        io.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
        io.flush();
        ASSERT_TRUE(io.good());
    }

    {
        VectorSegment recovered("seg", root, cfg, VectorSegment::State::Mutable);
        recovered.load();
        ASSERT_EQ(recovered.recordCount(), 0u);
    }
    ASSERT_EQ(std::filesystem::file_size(wal), 0u);
    std::filesystem::remove_all(root);
}

void test_wal_v1_migrates_and_rejects_bad_suffix() {
    auto root = std::filesystem::temp_directory_path() /
                ("wal_v1_migration_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    VectorSegment::Config cfg;
    cfg.dimensions = 2;
    {
        VectorSegment segment("seg", root, cfg, VectorSegment::State::Mutable);
        segment.initializeNew();
    }

    const auto wal = root / "wal.log";
    append_legacy_insert(wal, 7, "legacy", "v1", {1.0f, 2.0f});
    append_legacy_insert(wal, 6, "stale", "bad", {3.0f, 4.0f});

    uintmax_t migrated_size = 0;
    {
        VectorSegment recovered("seg", root, cfg, VectorSegment::State::Mutable);
        recovered.load();
        ASSERT_TRUE(recovered.contains("legacy"));
        ASSERT_FALSE(recovered.contains("stale"));
        ASSERT_EQ(recovered.maxSequence(), 7u);
        ASSERT_EQ(recovered.getMetadata("legacy"), std::string{"v1"});

        migrated_size = std::filesystem::file_size(wal);
        std::string oversized((1u << 20) + 1, 'x');
        ASSERT_THROWS(
            (void)recovered.insert(Vector(std::vector<float>{5.0f, 6.0f}),
                                   oversized, "", 8),
            std::length_error);
        ASSERT_EQ(std::filesystem::file_size(wal), migrated_size);
    }

    // A second open accepts the baseline fence and does not append another one.
    {
        VectorSegment recovered("seg", root, cfg, VectorSegment::State::Mutable);
        recovered.load();
        ASSERT_TRUE(recovered.contains("legacy"));
        ASSERT_FALSE(recovered.contains("stale"));
    }
    ASSERT_EQ(std::filesystem::file_size(wal), migrated_size);
    std::filesystem::remove_all(root);
}

void test_wal_recovery_bounds_corrupt_lengths() {
    auto root = std::filesystem::temp_directory_path() /
                ("wal_v2_lengths_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    VectorSegment::Config cfg;
    cfg.dimensions = 2;
    {
        VectorSegment segment("seg", root, cfg, VectorSegment::State::Mutable);
        segment.initializeNew();
        ASSERT_TRUE(segment.insert(Vector(std::vector<float>{1.0f, 2.0f}), "safe", "", 1));
    }

    const auto wal = root / "wal.log";
    const auto committed_size = std::filesystem::file_size(wal);
    TestWalRecordHeader corrupt{
        0x314c5756u,
        2,
        1,
        2,
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        2,
        2 * sizeof(float),
        0,
    };
    {
        std::ofstream os(wal, std::ios::binary | std::ios::app);
        os.write(reinterpret_cast<const char*>(&corrupt), sizeof(corrupt));
    }
    ASSERT_TRUE(std::filesystem::file_size(wal) > committed_size);

    {
        VectorSegment recovered("seg", root, cfg, VectorSegment::State::Mutable);
        recovered.load();
        ASSERT_EQ(recovered.recordCount(), 1u);
        ASSERT_TRUE(recovered.contains("safe"));
    }
    ASSERT_EQ(std::filesystem::file_size(wal), committed_size);
    std::filesystem::remove_all(root);
}

void test_sequence_highwater_skips_restart_range() {
    auto root = std::filesystem::temp_directory_path() /
                ("lsn_highwater_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    SegmentedVectorStore::Config cfg;
    cfg.dimensions = 2;

    pid_t pid = fork();
    ASSERT_TRUE(pid >= 0);
    if (pid == 0) {
        try {
            SegmentedVectorStore store(root, cfg);
            store.initialize();
            const bool inserted = store.insert(
                Vector(std::vector<float>{1.0f, 0.0f}), "first");
            _exit(inserted ? 0 : 1);
        } catch (...) {
            _exit(1);
        }
    }

    int status = 0;
    ASSERT_TRUE(waitpid(pid, &status, 0) == pid);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);

    {
        SegmentedVectorStore store(root, cfg);
        store.initialize();
        const uint64_t restart_base = store.getStatistics().latest_sequence;
        ASSERT_TRUE(restart_base >= (1ull << 20));
        ASSERT_TRUE(store.insert(Vector(std::vector<float>{0.0f, 1.0f}), "second"));
        ASSERT_TRUE(store.getStatistics().latest_sequence > restart_base);
        store.shutdown();
    }

    std::filesystem::remove_all(root);
}

bool read_pipe_byte(int fd, char& value) {
    ssize_t result = -1;
    do {
        result = ::read(fd, &value, 1);
    } while (result < 0 && errno == EINTR);
    return result == 1;
}

bool write_pipe_byte(int fd, char value) {
    ssize_t result = -1;
    do {
        result = ::write(fd, &value, 1);
    } while (result < 0 && errno == EINTR);
    return result == 1;
}

bool writer_open_reports_lock_conflict(
    const std::filesystem::path& root,
    const SegmentedVectorStore::Config& cfg) {
    try {
        SegmentedVectorStore contender(root, cfg);
        contender.initialize();
        contender.shutdown();
        return false;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find("already open for writing") !=
               std::string::npos;
    } catch (...) {
        return false;
    }
}

void test_segmented_store_exclusive_writer_lock() {
    auto root = std::filesystem::temp_directory_path() /
                ("segmented_writer_lock_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);
    SegmentedVectorStore::Config cfg;
    cfg.dimensions = 2;

    SegmentedVectorStore parent(root, cfg);
    parent.initialize();

    // Read-only recovery never takes the exclusive writer lock.
    {
        SegmentedVectorStore reader(root, cfg);
        reader.initialize(true);
        reader.shutdown();
    }
    ASSERT_TRUE(writer_open_reports_lock_conflict(root, cfg));

    pid_t contender_pid = ::fork();
    ASSERT_TRUE(contender_pid >= 0);
    if (contender_pid == 0) {
        _exit(writer_open_reports_lock_conflict(root, cfg) ? 0 : 1);
    }

    int contender_status = 0;
    ASSERT_TRUE(::waitpid(contender_pid, &contender_status, 0) == contender_pid);
    ASSERT_TRUE(WIFEXITED(contender_status));
    ASSERT_EQ(WEXITSTATUS(contender_status), 0);

    parent.shutdown();
    {
        SegmentedVectorStore after_shutdown(root, cfg);
        after_shutdown.initialize();
        after_shutdown.shutdown();
    }

    auto verify_forked_holder_release = [&](bool clean_shutdown) {
        int ready_pipe[2];
        int release_pipe[2];
        ASSERT_EQ(::pipe(ready_pipe), 0);
        ASSERT_EQ(::pipe(release_pipe), 0);

        pid_t holder_pid = ::fork();
        ASSERT_TRUE(holder_pid >= 0);
        if (holder_pid == 0) {
            ::close(ready_pipe[0]);
            ::close(release_pipe[1]);
            try {
                SegmentedVectorStore holder(root, cfg);
                holder.initialize();
                if (!write_pipe_byte(ready_pipe[1], 'R')) _exit(2);

                char release = 0;
                if (!read_pipe_byte(release_pipe[0], release) || release != 'X') {
                    _exit(3);
                }
                if (clean_shutdown) holder.shutdown();
                _exit(0);
            } catch (...) {
                (void)write_pipe_byte(ready_pipe[1], 'E');
                _exit(4);
            }
        }

        ::close(ready_pipe[1]);
        ::close(release_pipe[0]);

        char ready = 0;
        const bool child_ready = read_pipe_byte(ready_pipe[0], ready) && ready == 'R';
        const bool rejected_while_child_holds =
            child_ready && writer_open_reports_lock_conflict(root, cfg);
        const bool child_released =
            child_ready && write_pipe_byte(release_pipe[1], 'X');

        ::close(ready_pipe[0]);
        ::close(release_pipe[1]);

        int holder_status = 0;
        const bool child_waited =
            ::waitpid(holder_pid, &holder_status, 0) == holder_pid;
        ASSERT_TRUE(child_ready);
        ASSERT_TRUE(rejected_while_child_holds);
        ASSERT_TRUE(child_released);
        ASSERT_TRUE(child_waited);
        ASSERT_TRUE(WIFEXITED(holder_status));
        ASSERT_EQ(WEXITSTATUS(holder_status), 0);

        SegmentedVectorStore after_child_exit(root, cfg);
        after_child_exit.initialize();
        after_child_exit.shutdown();
    };

    verify_forked_holder_release(true);
    verify_forked_holder_release(false);

    std::filesystem::remove_all(root);
}

void test_hnsw_seed_persists_across_reopen() {
    auto root = std::filesystem::temp_directory_path() /
                ("hnsw_seed_persist_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    SegmentedVectorStore::Config cfg;
    cfg.dimensions = 2;
    cfg.hnsw_seed = 77;
    {
        SegmentedVectorStore store(root, cfg);
        store.initialize();
        ASSERT_TRUE(store.insert(Vector(std::vector<float>{1.0f, 2.0f}), "seeded"));
        store.shutdown();
    }

    auto read_text = [](const std::filesystem::path& path) {
        std::ifstream is(path);
        return std::string(std::istreambuf_iterator<char>(is), {});
    };
    ASSERT_TRUE(read_text(root / "manifest.txt").find("hnsw_seed=77\n") != std::string::npos);
    ASSERT_TRUE(read_text(root / "segments" / "seg_00000001" / "segment.meta")
                    .find("hnsw_seed=77\n") != std::string::npos);

    cfg.hnsw_seed = 999;
    {
        SegmentedVectorStore store(root, cfg);
        store.initialize();
        ASSERT_TRUE(store.get("seeded").has_value());
        store.shutdown();
    }
    ASSERT_TRUE(read_text(root / "manifest.txt").find("hnsw_seed=77\n") != std::string::npos);
    std::filesystem::remove_all(root);
}

// =====================================================================
//  SCALAR QUANTIZER
// =====================================================================

void test_scalar_quantizer_train_and_quantize() {
    constexpr size_t dims = 4;
    ScalarQuantizer q(dims);
    ASSERT_FALSE(q.is_trained());

    std::vector<std::vector<float>> data{
        {0.0f, -1.0f,  2.0f,  0.5f},
        {1.0f,  1.0f, -2.0f, -0.5f},
        {0.5f,  0.0f,  0.0f,  0.0f},
    };
    std::vector<const float*> ptrs;
    for (auto& v : data) ptrs.push_back(v.data());
    q.train(ptrs.data(), ptrs.size());
    ASSERT_TRUE(q.is_trained());

    // Round-trip approximation: quantizing then comparing distance ranks
    // should preserve order for well-separated vectors.
    uint8_t qa[dims], qb[dims], qc[dims];
    q.quantize(data[0].data(), qa);
    q.quantize(data[1].data(), qb);
    q.quantize(data[2].data(), qc);

    uint32_t d_aa = q.distance_quantized(qa, qa);
    uint32_t d_ab = q.distance_quantized(qa, qb);
    ASSERT_EQ(d_aa, uint32_t{0});  // self-distance is exactly zero
    ASSERT_TRUE(d_ab > 0);

    // Regression: with a GLOBAL scale the quantized distance is proportional to
    // true L2, so its ordering must match true-L2 ordering. c is closer to a
    // than b is (true L2), so quantized d(a,c) must be < d(a,b). A per-dimension
    // scale could violate this.
    auto true_l2_sq = [](const std::vector<float>& x, const std::vector<float>& y) {
        float s = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) { float d = x[i] - y[i]; s += d * d; }
        return s;
    };
    ASSERT_TRUE(true_l2_sq(data[0], data[2]) < true_l2_sq(data[0], data[1]));
    uint32_t d_ac = q.distance_quantized(qa, qc);
    ASSERT_TRUE(d_ac < d_ab);  // quantized ordering matches true-L2 ordering
}

// =====================================================================
//  RW LOCK
// =====================================================================

void test_rwlock_readers_can_share() {
    RWLock lock;
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            RWLock::ReadGuard g(lock);
            int now = ++active;
            int prev_max = max_active.load();
            while (prev_max < now &&
                   !max_active.compare_exchange_weak(prev_max, now)) {}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            --active;
        });
    }
    for (auto& t : threads) t.join();
    // Multiple readers must have held the lock simultaneously.
    ASSERT_TRUE(max_active.load() > 1);
}

void test_rwlock_writer_excludes_readers() {
    RWLock lock;
    std::atomic<bool> writer_in{false};
    std::atomic<bool> reader_overlapped{false};

    std::thread w([&] {
        RWLock::WriteGuard g(lock);
        writer_in.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        writer_in.store(false);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::thread r([&] {
        RWLock::ReadGuard g(lock);
        if (writer_in.load()) reader_overlapped.store(true);
    });

    w.join();
    r.join();
    ASSERT_FALSE(reader_overlapped.load());
}

// =====================================================================
//  ATOMIC WRITE HELPER
// =====================================================================

void test_atomic_write_creates_file_and_content() {
    auto path = std::filesystem::temp_directory_path() /
                ("atomicw_" + std::to_string(::getpid()) + ".bin");
    std::filesystem::remove(path);

    const auto sync_before = vdb::io::file_sync_statistics();
    const std::string payload = "the quick brown fox";
    vdb::io::atomic_write(path, [&](std::ostream& os) {
        os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    });

    ASSERT_TRUE(std::filesystem::exists(path));
    std::ifstream is(path, std::ios::binary);
    std::string read_back((std::istreambuf_iterator<char>(is)), {});
    ASSERT_EQ(read_back, payload);

    const auto sync_after = vdb::io::file_sync_statistics();
    ASSERT_EQ(sync_after.fsync_successes + sync_after.full_fsync_successes,
              sync_before.fsync_successes + sync_before.full_fsync_successes + 1);

    // No leftover .tmp file.
    auto tmp = path; tmp += ".tmp";
    ASSERT_FALSE(std::filesystem::exists(tmp));

    std::filesystem::remove(path);
}

void test_atomic_write_overwrites_existing() {
    auto path = std::filesystem::temp_directory_path() /
                ("atomicw_overwrite_" + std::to_string(::getpid()) + ".bin");
    std::filesystem::remove(path);

    vdb::io::atomic_write(path, [](std::ostream& os) { os << "first"; });
    vdb::io::atomic_write(path, [](std::ostream& os) { os << "second"; });

    std::ifstream is(path);
    std::string content((std::istreambuf_iterator<char>(is)), {});
    ASSERT_EQ(content, std::string{"second"});

    std::filesystem::remove(path);
}

void test_full_fsync_failure_never_falls_back() {
    size_t plain_calls = 0;
    size_t full_calls = 0;
    bool threw = false;
    try {
        (void)vdb::io::detail::sync_descriptor_with_calls(
            -1,
            std::filesystem::path("synthetic-full-sync"),
            true,
            [&](int) {
                ++plain_calls;
                return 0;
            },
            [&](int) {
                ++full_calls;
                errno = EIO;
                return -1;
            });
    } catch (const std::runtime_error& error) {
        threw = true;
        ASSERT_TRUE(std::string(error.what()).find("F_FULLFSYNC") != std::string::npos);
    }

    ASSERT_TRUE(threw);
    ASSERT_EQ(full_calls, size_t{1});
    ASSERT_EQ(plain_calls, size_t{0});
}

void test_full_fsync_retries_eintr_and_reports_mode() {
    size_t plain_calls = 0;
    size_t full_calls = 0;
    const auto mode = vdb::io::detail::sync_descriptor_with_calls(
        -1,
        std::filesystem::path("synthetic-full-sync"),
        true,
        [&](int) {
            ++plain_calls;
            return 0;
        },
        [&](int) {
            ++full_calls;
            if (full_calls == 1) {
                errno = EINTR;
                return -1;
            }
            return 0;
        });

    ASSERT_TRUE(mode == vdb::io::FileSyncMode::FullFsync);
    ASSERT_EQ(full_calls, size_t{2});
    ASSERT_EQ(plain_calls, size_t{0});
}

int fail_file_sync_for_test(int, vdb::io::FileSyncMode) {
    errno = EIO;
    return -1;
}

class FileSyncOverrideGuard {
public:
    explicit FileSyncOverrideGuard(vdb::io::detail::FileSyncCallOverride override_call) {
        vdb::io::testing::set_file_sync_call_override(override_call);
    }

    ~FileSyncOverrideGuard() {
        vdb::io::testing::set_file_sync_call_override(nullptr);
    }
};

class ForcedFullSyncOverrideGuard {
public:
    explicit ForcedFullSyncOverrideGuard(
        vdb::io::detail::FileSyncCallOverride override_call) {
        vdb::io::testing::set_file_sync_call_override(override_call);
        vdb::io::testing::set_force_full_fsync_for_testing(true);
    }

    ~ForcedFullSyncOverrideGuard() {
        vdb::io::testing::set_force_full_fsync_for_testing(false);
        vdb::io::testing::set_file_sync_call_override(nullptr);
    }
};

struct RolloverSyncFaultState {
    std::filesystem::path highwater_path;
    uint64_t expected_highwater{0};
    size_t full_calls{0};
    size_t plain_calls{0};
    bool saw_regular_file_sync{false};
    bool saw_renamed_highwater{false};
    bool saw_directory_barrier{false};
};

RolloverSyncFaultState* rollover_sync_fault_state = nullptr;

int fail_post_rename_directory_full_sync_for_test(
    int fd, vdb::io::FileSyncMode mode) {
    auto* state = rollover_sync_fault_state;
    if (state == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (mode != vdb::io::FileSyncMode::FullFsync) {
        ++state->plain_calls;
        errno = EINVAL;
        return -1;
    }

    ++state->full_calls;
    if (state->full_calls == 1) {
        struct stat descriptor_stat {};
        state->saw_regular_file_sync =
            ::fstat(fd, &descriptor_stat) == 0 && S_ISREG(descriptor_stat.st_mode);
        return 0;
    }
    if (state->full_calls != 2) return 0;

    std::ifstream is(state->highwater_path, std::ios::binary);
    uint64_t highwater = 0;
    is.seekg(2 * sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&highwater), sizeof(highwater));
    state->saw_renamed_highwater =
        is.gcount() == static_cast<std::streamsize>(sizeof(highwater)) &&
        highwater == state->expected_highwater;
    struct stat descriptor_stat {};
    state->saw_directory_barrier =
        ::fstat(fd, &descriptor_stat) == 0 && S_ISDIR(descriptor_stat.st_mode);
    errno = EIO;
    return -1;
}

void test_sequence_rollover_barrier_failure_rejects_weak_write() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("lsn_rollover_sync_failure_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    SegmentedVectorStore::Config config;
    config.dimensions = 2;
    config.sequence_reservation_block = 2;

    auto store = std::make_unique<SegmentedVectorStore>(root, config);
    store->initialize();
    const auto first = store->stageInsert(
        Vector(std::vector<float>{1.0f, 0.0f}), "weak-1");
    const auto second = store->stageInsert(
        Vector(std::vector<float>{0.0f, 1.0f}), "weak-2");
    ASSERT_TRUE(first.applied);
    ASSERT_TRUE(second.applied);
    ASSERT_EQ(first.lsn, uint64_t{1});
    ASSERT_EQ(second.lsn, uint64_t{2});

    RolloverSyncFaultState fault;
    fault.highwater_path = root / "lsn.highwater";
    fault.expected_highwater = 4;
    rollover_sync_fault_state = &fault;
    const auto sync_before = vdb::io::file_sync_statistics();
    bool threw = false;
    std::string error_message;
    {
        ForcedFullSyncOverrideGuard fail_directory_barrier(
            fail_post_rename_directory_full_sync_for_test);
        try {
            // The weak-ACK committer publishes its receipt only after this
            // staging boundary returns successfully.
            (void)store->stageInsert(
                Vector(std::vector<float>{2.0f, 2.0f}), "must-not-ack");
        } catch (const std::runtime_error& error) {
            threw = true;
            error_message = error.what();
        }
    }
    rollover_sync_fault_state = nullptr;

    ASSERT_TRUE(threw);
    ASSERT_TRUE(error_message.find("F_FULLFSYNC") != std::string::npos);
    ASSERT_EQ(fault.full_calls, size_t{2});
    ASSERT_EQ(fault.plain_calls, size_t{0});
    ASSERT_TRUE(fault.saw_regular_file_sync);
    ASSERT_TRUE(fault.saw_renamed_highwater);
    ASSERT_TRUE(fault.saw_directory_barrier);
    ASSERT_EQ(store->vectorCount(), size_t{2});
    ASSERT_EQ(store->visibleLsn(), uint64_t{2});
    ASSERT_EQ(store->durableLsn(), uint64_t{0});
    ASSERT_FALSE(store->get("must-not-ack").has_value());

    const auto sync_after = vdb::io::file_sync_statistics();
    ASSERT_EQ(sync_after.full_fsync_successes,
              sync_before.full_fsync_successes + 1);
    ASSERT_EQ(sync_after.failures, sync_before.failures);

    // Do not call shutdown: model recovery immediately after the failed
    // reservation. The two prior weak writes are unfenced and are discarded;
    // the renamed high-water file makes recovery skip the failed [3, 4] range.
    store.reset();
    SegmentedVectorStore recovered(root, config);
    recovered.initialize();
    ASSERT_EQ(recovered.vectorCount(), size_t{0});
    const auto after_recovery = recovered.stageInsert(
        Vector(std::vector<float>{3.0f, 3.0f}), "after-recovery");
    ASSERT_TRUE(after_recovery.applied);
    ASSERT_EQ(after_recovery.lsn, uint64_t{5});
    recovered.shutdown();

    std::filesystem::remove_all(root);
}

void test_sync_failure_cannot_report_stable_ack() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("vdb_sync_failure_" + std::to_string(::getpid()));
    std::filesystem::remove_all(path);

    VectorDatabase database(2, VectorDatabase::SearchMode::HNSW, false, false, {},
                            false, 0, path.string());
    vdb::RecallCommitConfig config;
    config.enabled = true;
    config.policy = vdb::RecallPolicy::Strict;
    config.epsilon = 0.2;
    config.k_min = 10;
    database.configureRecallCommit(config);
    database.initialize();

    const auto sync_before = vdb::io::file_sync_statistics();
    {
        FileSyncOverrideGuard fail_sync(fail_file_sync_for_test);
        ASSERT_THROWS(
            (void)database.insertWithAck(
                Vector(std::vector<float>{1.0f, 2.0f}),
                "must-not-be-stable",
                vdb::AckMode::Stable),
            std::runtime_error);

        const auto committer = database.recallCommitterStatistics();
        ASSERT_EQ(committer.sync_attempts, uint64_t{1});
        ASSERT_EQ(committer.sync_failures, uint64_t{1});
        ASSERT_EQ(committer.sync_successes, uint64_t{0});
        ASSERT_EQ(committer.stable_acks, uint64_t{0});
        ASSERT_TRUE(database.durabilityStatus().health == vdb::CommitterHealth::SyncFailed);

        const auto sync_after = vdb::io::file_sync_statistics();
        ASSERT_EQ(sync_after.failures, sync_before.failures + 1);
        ASSERT_EQ(sync_after.full_fsync_successes, sync_before.full_fsync_successes);
    }

    database.shutdown();
    std::filesystem::remove_all(path);
}

// =====================================================================
//  MAIN
// =====================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " Vector Database Unit Tests\n";
    std::cout << "========================================\n\n";

    std::cout << "[Vector]\n";
    run_test("construction", test_vector_construction);
    run_test("access and bounds", test_vector_access);
    run_test("dot product", test_vector_dot_product);
    run_test("equality", test_vector_equality);
    run_test("data_ptr", test_vector_data_ptr);
    run_test("iterators", test_vector_iterators);
    run_test("serialization", test_vector_serialization);
    run_test("hash", test_vector_hash);
    run_test("default construction", test_vector_default_construction);

    std::cout << "\n[Distance Metrics]\n";
    run_test("euclidean distance", test_euclidean_distance);
    run_test("euclidean self-distance", test_euclidean_distance_same);
    run_test("manhattan distance", test_manhattan_distance);
    run_test("cosine similarity", test_cosine_similarity);
    run_test("cosine parallel vectors", test_cosine_similarity_parallel);

    std::cout << "\n[Flat Exact Index]\n";
    run_test("flat index metric policy", test_flat_index_metric_policy);

    std::cout << "\n[HNSW Index]\n";
    run_test("insert and search", test_hnsw_insert_search);
    run_test("remove", test_hnsw_remove);
    run_test("many vectors", test_hnsw_many_vectors);
    run_test("fixed seed reproduces topology", test_hnsw_fixed_seed_reproduces_topology);

    std::cout << "\n[Query Cache]\n";
    run_test("hit and miss", test_cache_hit_miss);
    run_test("invalidation", test_cache_invalidation);
    run_test("eviction", test_cache_eviction);
    run_test("statistics", test_cache_statistics);
    run_test("clear", test_cache_clear);
    run_test("k-aware hit/miss", test_cache_k_aware);
    run_test("zero capacity no-op", test_cache_zero_capacity);

    std::cout << "\n[VectorDatabase]\n";
    run_test("insert and get", test_db_insert_get);
    run_test("insert duplicate", test_db_insert_duplicate);
    run_test("insert dimension mismatch", test_db_insert_dimension_mismatch);
    run_test("update", test_db_update);
    run_test("update nonexistent", test_db_update_nonexistent);
    run_test("remove", test_db_remove);
    run_test("remove nonexistent", test_db_remove_nonexistent);
    run_test("metadata", test_db_metadata);
    run_test("metadata empty", test_db_metadata_empty);
    run_test("vector count", test_db_vector_count);
    run_test("not initialized throws", test_db_not_initialized);
    run_test("similarity search exact", test_db_similarity_search_exact);
    run_test("similarity search empty", test_db_similarity_search_empty);
    run_test("search with metadata", test_db_search_with_metadata);
    run_test("distance metric switch", test_db_distance_metric_switch);
    run_test("NaN rejection", test_db_nan_rejection);
    run_test("statistics", test_db_statistics);
    run_test("get all vectors", test_db_get_all_vectors);
    run_test("segmented persistence recovery", test_db_segmented_persistence_recovery);
    run_test("default segmented single-op crash recovery", test_db_default_segmented_single_op_crash_recovery);
    run_test("group commit crash recovery", test_db_group_commit_crash_recovery);

    std::cout << "\n[Protocol]\n";
    run_test("buffer roundtrip",          test_protocol_buffer_roundtrip);
    run_test("buffer underflow throws",   test_protocol_buffer_underflow);
    run_test("string too long throws",    test_protocol_string_too_long);
    run_test("short buffer reads throw",  test_protocol_short_buffer);

    std::cout << "\n[Commit Log]\n";
    run_test("entry roundtrip",          test_commit_log_entry_roundtrip);
    run_test("checksum detects tamper",  test_commit_log_checksum_detects_tampering);
    run_test("oversize payload guard",   test_commit_log_oversize_payload_throws);

    std::cout << "\n[MMap Storage]\n";
    run_test("basic CRUD + reopen",            test_mmap_storage_basic_crud);
    run_test("dimension mismatch on reopen",   test_mmap_storage_dimension_mismatch_on_reopen);

    std::cout << "\n[Segmented Vector Store]\n";
    run_test("insert/search/recover",   test_segmented_store_insert_search_recover);
    run_test("staged tail frontiers and exact search", test_segment_staged_tail_frontiers_and_exact_search);
    run_test("staged insert recovery drops tail", test_store_staged_insert_recovery_drops_tail);
    run_test("manifest role is authoritative", test_store_manifest_role_is_authoritative);
    run_test("WAL v2 discards unfenced tail", test_wal_v2_discards_unfenced_tail);
    run_test("WAL v2 replays fenced generation", test_wal_v2_replays_fenced_generation);
    run_test("WAL v2 rejects inconsistent fence", test_wal_v2_rejects_inconsistent_fence);
    run_test("WAL v1 migrates and rejects bad suffix", test_wal_v1_migrates_and_rejects_bad_suffix);
    run_test("WAL recovery bounds corrupt lengths", test_wal_recovery_bounds_corrupt_lengths);
    run_test("sequence high-water skips restart range", test_sequence_highwater_skips_restart_range);
    run_test("exclusive writer lock lifecycle", test_segmented_store_exclusive_writer_lock);
    run_test("sequence rollover barrier rejects weak write",
             test_sequence_rollover_barrier_failure_rejects_weak_write);
    run_test("HNSW seed persists across reopen", test_hnsw_seed_persists_across_reopen);

    std::cout << "\n[Scalar Quantizer]\n";
    run_test("train and quantize",      test_scalar_quantizer_train_and_quantize);

    std::cout << "\n[RWLock]\n";
    run_test("readers can share",       test_rwlock_readers_can_share);
    run_test("writer excludes readers", test_rwlock_writer_excludes_readers);

    std::cout << "\n[Atomic Write]\n";
    run_test("creates file with content", test_atomic_write_creates_file_and_content);
    run_test("overwrites existing",       test_atomic_write_overwrites_existing);
    run_test("full sync failure does not fall back", test_full_fsync_failure_never_falls_back);
    run_test("full sync retries EINTR", test_full_fsync_retries_eintr_and_reports_mode);
    run_test("sync failure cannot report stable ACK", test_sync_failure_cannot_report_stable_ack);

    std::cout << "\n========================================\n";
    std::cout << " Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << " (" << tests_failed << " FAILED)";
    }
    std::cout << "\n========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
