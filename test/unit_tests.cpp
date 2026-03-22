// test/unit_tests.cpp
// Lightweight unit test framework (no external dependencies)

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../src/core/vector.hpp"
#include "../src/core/kd_tree.hpp"
#include "../src/core/vector_database.hpp"
#include "../src/algorithms/lsh_index.hpp"
#include "../src/algorithms/hnsw_index.hpp"
#include "../src/features/query_cache.hpp"
#include "../src/utils/distance_metrics.hpp"

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
    ASSERT_NEAR(dot, 32.0f, 1e-4f); // 1*4 + 2*5 + 3*6 = 32

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
    // Different vectors should (almost certainly) have different hashes
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
}

void test_euclidean_distance_same() {
    EuclideanDistance metric;
    Vector v(std::vector<float>{1.0f, 2.0f, 3.0f});
    ASSERT_NEAR(metric.distance(v, v), 0.0f, 1e-6f);
}

void test_manhattan_distance() {
    ManhattanDistance metric;
    Vector v1(std::vector<float>{1.0f, 2.0f, 3.0f});
    Vector v2(std::vector<float>{4.0f, 6.0f, 3.0f});
    float d = metric.distance(v1, v2);
    ASSERT_NEAR(d, 7.0f, 1e-4f); // |3| + |4| + |0| = 7
}

void test_cosine_similarity() {
    CosineSimilarity metric;
    Vector v1(std::vector<float>{1.0f, 0.0f});
    Vector v2(std::vector<float>{0.0f, 1.0f});
    float d = metric.distance(v1, v2);
    // Perpendicular vectors: cosine similarity = 0, cosine distance = 1
    ASSERT_NEAR(d, 1.0f, 1e-4f);

    // Identical direction
    Vector v3(std::vector<float>{2.0f, 0.0f});
    float d2 = metric.distance(v1, v3);
    ASSERT_NEAR(d2, 0.0f, 1e-4f);
}

void test_cosine_similarity_parallel() {
    CosineSimilarity metric;
    Vector v1(std::vector<float>{1.0f, 1.0f});
    Vector v2(std::vector<float>{2.0f, 2.0f});
    float d = metric.distance(v1, v2);
    ASSERT_NEAR(d, 0.0f, 1e-4f); // Same direction = 0 distance
}

// =====================================================================
//  KD-TREE TESTS
// =====================================================================

void test_kd_tree_insert_search() {
    auto metric = std::make_shared<EuclideanDistance>();
    KDTree tree(3, metric);

    Vector v1(std::vector<float>{1.0f, 0.0f, 0.0f});
    Vector v2(std::vector<float>{0.0f, 1.0f, 0.0f});
    Vector v3(std::vector<float>{0.0f, 0.0f, 1.0f});

    tree.insert(v1, "a");
    tree.insert(v2, "b");
    tree.insert(v3, "c");

    Vector query(std::vector<float>{0.9f, 0.1f, 0.0f});
    auto results = tree.nearestNeighbors(query, 1);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].first, "a");
}

void test_kd_tree_k_nearest() {
    auto metric = std::make_shared<EuclideanDistance>();
    KDTree tree(2, metric);

    tree.insert(Vector(std::vector<float>{0.0f, 0.0f}), "origin");
    tree.insert(Vector(std::vector<float>{1.0f, 0.0f}), "right");
    tree.insert(Vector(std::vector<float>{10.0f, 10.0f}), "far");

    Vector query(std::vector<float>{0.5f, 0.0f});
    auto results = tree.nearestNeighbors(query, 2);
    ASSERT_EQ(results.size(), 2u);
    // Closest should be "origin" (0.5) or "right" (0.5)
    // Both are equidistant, just check we get 2 results
    ASSERT_TRUE(results[0].second <= results[1].second);
}

void test_kd_tree_remove() {
    auto metric = std::make_shared<EuclideanDistance>();
    KDTree tree(2, metric);

    tree.insert(Vector(std::vector<float>{0.0f, 0.0f}), "a");
    tree.insert(Vector(std::vector<float>{1.0f, 1.0f}), "b");
    tree.insert(Vector(std::vector<float>{2.0f, 2.0f}), "c");

    tree.remove("b");

    Vector query(std::vector<float>{1.0f, 1.0f});
    auto results = tree.nearestNeighbors(query, 3);
    // "b" should not appear in results
    for (const auto& [key, dist] : results) {
        ASSERT_TRUE(key != "b");
    }
}

void test_kd_tree_empty() {
    auto metric = std::make_shared<EuclideanDistance>();
    KDTree tree(2, metric);

    Vector query(std::vector<float>{0.0f, 0.0f});
    auto results = tree.nearestNeighbors(query, 5);
    ASSERT_EQ(results.size(), 0u);
}

// =====================================================================
//  LSH INDEX TESTS
// =====================================================================

void test_lsh_insert_search() {
    auto metric = std::make_shared<EuclideanDistance>();
    LSHIndex lsh(4, 5, 4, metric);

    Vector v1(std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    Vector v2(std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});
    Vector v3(std::vector<float>{1.0f, 0.1f, 0.0f, 0.0f});

    lsh.insert(v1, "a");
    lsh.insert(v2, "b");
    lsh.insert(v3, "c");

    Vector query(std::vector<float>{1.0f, 0.05f, 0.0f, 0.0f});
    auto results = lsh.search(query, 2);
    // Should find at least 1 result (LSH is approximate, may not always return all)
    ASSERT_TRUE(results.size() >= 1);
}

void test_lsh_remove() {
    auto metric = std::make_shared<EuclideanDistance>();
    LSHIndex lsh(2, 5, 4, metric);

    lsh.insert(Vector(std::vector<float>{1.0f, 0.0f}), "a");
    lsh.insert(Vector(std::vector<float>{0.0f, 1.0f}), "b");
    lsh.remove("a");

    Vector query(std::vector<float>{1.0f, 0.0f});
    auto results = lsh.search(query, 10);
    for (const auto& [key, dist] : results) {
        ASSERT_TRUE(key != "a");
    }
}

// =====================================================================
//  HNSW INDEX TESTS
// =====================================================================

void test_hnsw_insert_search() {
    auto metric = std::make_shared<EuclideanDistance>();
    HNSWIndex hnsw(3, 8, 50, 50, metric);

    hnsw.insert(Vector(std::vector<float>{1.0f, 0.0f, 0.0f}), "a");
    hnsw.insert(Vector(std::vector<float>{0.0f, 1.0f, 0.0f}), "b");
    hnsw.insert(Vector(std::vector<float>{0.0f, 0.0f, 1.0f}), "c");
    hnsw.insert(Vector(std::vector<float>{1.0f, 1.0f, 0.0f}), "d");

    Vector query(std::vector<float>{1.0f, 0.9f, 0.0f});
    auto results = hnsw.search(query, 1);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].first, "d");
}

void test_hnsw_remove() {
    auto metric = std::make_shared<EuclideanDistance>();
    HNSWIndex hnsw(2, 8, 50, 50, metric);

    hnsw.insert(Vector(std::vector<float>{0.0f, 0.0f}), "origin");
    hnsw.insert(Vector(std::vector<float>{1.0f, 0.0f}), "right");
    hnsw.insert(Vector(std::vector<float>{0.0f, 1.0f}), "up");

    hnsw.remove("origin");

    Vector query(std::vector<float>{0.0f, 0.0f});
    auto results = hnsw.search(query, 10);
    for (const auto& [key, dist] : results) {
        ASSERT_TRUE(key != "origin");
    }
}

void test_hnsw_many_vectors() {
    auto metric = std::make_shared<EuclideanDistance>();
    HNSWIndex hnsw(4, 8, 50, 50, metric);

    // Insert 100 vectors
    for (int i = 0; i < 100; i++) {
        float f = static_cast<float>(i);
        Vector v(std::vector<float>{f, f * 0.5f, f * 0.25f, f * 0.1f});
        hnsw.insert(v, "v" + std::to_string(i));
    }

    ASSERT_EQ(hnsw.size(), 100u);

    // Search for a vector close to v50
    Vector query(std::vector<float>{50.0f, 25.0f, 12.5f, 5.0f});
    auto results = hnsw.search(query, 3);
    ASSERT_TRUE(results.size() >= 1);
    ASSERT_EQ(results[0].first, "v50");
}

// =====================================================================
//  QUERY CACHE TESTS
// =====================================================================

void test_cache_hit_miss() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f, 2.0f});

    std::vector<std::pair<std::string, float>> results;
    ASSERT_FALSE(cache.get(q, results)); // miss

    std::vector<std::pair<std::string, float>> data = {{"a", 0.5f}, {"b", 1.0f}};
    cache.put(q, data);

    ASSERT_TRUE(cache.get(q, results)); // hit
    ASSERT_EQ(results.size(), 2u);
    ASSERT_EQ(results[0].first, "a");
}

void test_cache_invalidation() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f, 2.0f});

    std::vector<std::pair<std::string, float>> data = {{"a", 0.5f}};
    cache.put(q, data);

    cache.invalidate(); // O(1) generation bump

    std::vector<std::pair<std::string, float>> results;
    ASSERT_FALSE(cache.get(q, results)); // stale entry
}

void test_cache_eviction() {
    QueryCache cache(3); // capacity 3

    for (int i = 0; i < 5; i++) {
        Vector q(std::vector<float>{static_cast<float>(i), 0.0f});
        cache.put(q, {{"x", static_cast<float>(i)}});
    }

    auto stats = cache.getStatistics();
    ASSERT_TRUE(stats.current_size <= 3);
}

void test_cache_statistics() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f});

    std::vector<std::pair<std::string, float>> out;
    cache.get(q, out); // miss
    cache.put(q, {{"a", 1.0f}});
    cache.get(q, out); // hit

    auto stats = cache.getStatistics();
    ASSERT_EQ(stats.hits, 1u);
    ASSERT_EQ(stats.misses, 1u);
    ASSERT_NEAR(stats.hit_rate(), 0.5, 1e-6);
}

void test_cache_clear() {
    QueryCache cache(10);
    Vector q(std::vector<float>{1.0f});
    cache.put(q, {{"a", 1.0f}});

    cache.clear();
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
    ASSERT_FALSE(db.insert(v, "k")); // duplicate
}

void test_db_insert_dimension_mismatch() {
    VectorDatabase db(3);
    db.initialize();

    Vector v(std::vector<float>{1.0f, 2.0f}); // 2D, db expects 3D
    ASSERT_THROWS(db.insert(v, "k"), std::invalid_argument);
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
    ASSERT_THROWS(db.insert(v, "k"), std::runtime_error);
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
    ASSERT_EQ(results[0].first, "x"); // closest
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

    // Default: euclidean
    Vector query(std::vector<float>{0.9f, 0.1f});
    auto r1 = db.similaritySearch(query, 1);
    ASSERT_EQ(r1[0].first, "a");

    // Switch to manhattan
    db.setDistanceMetric(std::make_shared<ManhattanDistance>());
    auto r2 = db.similaritySearch(query, 1);
    ASSERT_EQ(r2[0].first, "a"); // still closest under manhattan
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

    std::cout << "\n[KD-Tree]\n";
    run_test("insert and search", test_kd_tree_insert_search);
    run_test("k-nearest neighbors", test_kd_tree_k_nearest);
    run_test("remove", test_kd_tree_remove);
    run_test("empty tree", test_kd_tree_empty);

    std::cout << "\n[LSH Index]\n";
    run_test("insert and search", test_lsh_insert_search);
    run_test("remove", test_lsh_remove);

    std::cout << "\n[HNSW Index]\n";
    run_test("insert and search", test_hnsw_insert_search);
    run_test("remove", test_hnsw_remove);
    run_test("many vectors", test_hnsw_many_vectors);

    std::cout << "\n[Query Cache]\n";
    run_test("hit and miss", test_cache_hit_miss);
    run_test("invalidation", test_cache_invalidation);
    run_test("eviction", test_cache_eviction);
    run_test("statistics", test_cache_statistics);
    run_test("clear", test_cache_clear);

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

    std::cout << "\n========================================\n";
    std::cout << " Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << " (" << tests_failed << " FAILED)";
    }
    std::cout << "\n========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
