#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../src/api/protocol.hpp"
#include "../src/api/tcp_server.hpp"
#include "../src/api/tcp_client.hpp"

// ── Minimal test harness ───────────────────────────────────────

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(expr)                                                    \
    do {                                                                     \
        if (!(expr)) {                                                       \
            std::cerr << "  FAIL: " #expr " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define ASSERT_EQ(a, b)                                                      \
    do {                                                                     \
        if ((a) != (b)) {                                                    \
            std::cerr << "  FAIL: " #a " == " #b " (got " << (a) << " vs " << (b) \
                      << ") (" << __FILE__ << ":" << __LINE__ << ")\n";      \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define ASSERT_NEAR(a, b, eps)                                               \
    do {                                                                     \
        if (std::abs((a) - (b)) > (eps)) {                                   \
            std::cerr << "  FAIL: " #a " ~= " #b " (got " << (a) << " vs " << (b) \
                      << ") (" << __FILE__ << ":" << __LINE__ << ")\n";      \
            return false;                                                    \
        }                                                                    \
    } while (0)

#define RUN_TEST(fn)                                                         \
    do {                                                                     \
        tests_run++;                                                         \
        std::cout << "  " << #fn << "... ";                                  \
        if (fn()) { tests_passed++; std::cout << "OK\n"; }                   \
        else      { tests_failed++; std::cout << "FAILED\n"; }               \
    } while (0)

// ── Test fixture: server + client ──────────────────────────────

static const int TEST_PORT = 19090;
static const size_t TEST_DIMS = 4;

struct TestFixture {
    TCPServer server;
    TCPClient client;

    TestFixture()
        : server(TEST_DIMS, "127.0.0.1", TEST_PORT, 2),
          client("127.0.0.1", TEST_PORT) {
        server.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        client.connect();
    }

    ~TestFixture() {
        client.disconnect();
        server.stop();
    }
};

// ── Tests ──────────────────────────────────────────────────────

bool test_ping() {
    TestFixture f;
    ASSERT_TRUE(f.client.ping());
    return true;
}

bool test_insert_and_get() {
    TestFixture f;
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(f.client.insert("vec1", v, "test metadata"));

    auto result = f.client.get("vec1");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->vector.size(), 4u);
    ASSERT_NEAR(result->vector[0], 1.0f, 1e-6f);
    ASSERT_NEAR(result->vector[3], 4.0f, 1e-6f);
    ASSERT_EQ(result->metadata, "test metadata");
    return true;
}

bool test_get_not_found() {
    TestFixture f;
    auto result = f.client.get("nonexistent");
    ASSERT_TRUE(!result.has_value());
    return true;
}

bool test_search() {
    TestFixture f;
    f.client.insert("a", {1.0f, 0.0f, 0.0f, 0.0f});
    f.client.insert("b", {0.0f, 1.0f, 0.0f, 0.0f});
    f.client.insert("c", {0.0f, 0.0f, 1.0f, 0.0f});

    auto results = f.client.search({1.0f, 0.1f, 0.0f, 0.0f}, 2);
    ASSERT_TRUE(results.size() >= 1);
    // "a" should be closest to (1, 0.1, 0, 0)
    ASSERT_EQ(results[0].key, "a");
    return true;
}

bool test_delete() {
    TestFixture f;
    f.client.insert("x", {1.0f, 1.0f, 1.0f, 1.0f});
    ASSERT_TRUE(f.client.remove("x"));

    auto result = f.client.get("x");
    ASSERT_TRUE(!result.has_value());
    return true;
}

bool test_delete_not_found() {
    TestFixture f;
    ASSERT_TRUE(!f.client.remove("ghost"));
    return true;
}

bool test_update() {
    TestFixture f;
    f.client.insert("u", {1.0f, 0.0f, 0.0f, 0.0f}, "original");
    ASSERT_TRUE(f.client.update("u", {0.0f, 1.0f, 0.0f, 0.0f}, "updated"));

    auto result = f.client.get("u");
    ASSERT_TRUE(result.has_value());
    ASSERT_NEAR(result->vector[0], 0.0f, 1e-6f);
    ASSERT_NEAR(result->vector[1], 1.0f, 1e-6f);
    ASSERT_EQ(result->metadata, "updated");
    return true;
}

bool test_batch_insert() {
    TestFixture f;
    std::vector<std::string> keys = {"b1", "b2", "b3"};
    std::vector<std::vector<float>> vecs = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f}
    };
    std::vector<std::string> meta = {"m1", "m2", "m3"};

    auto result = f.client.batch_insert(keys, vecs, meta);
    ASSERT_EQ(result.successful, 3u);
    ASSERT_EQ(result.failed, 0u);

    // Verify all were inserted
    for (const auto& key : keys) {
        auto got = f.client.get(key);
        ASSERT_TRUE(got.has_value());
    }
    return true;
}

bool test_batch_delete() {
    TestFixture f;
    f.client.insert("d1", {1.0f, 0.0f, 0.0f, 0.0f});
    f.client.insert("d2", {0.0f, 1.0f, 0.0f, 0.0f});
    f.client.insert("d3", {0.0f, 0.0f, 1.0f, 0.0f});

    auto result = f.client.batch_delete({"d1", "d2", "d3"});
    ASSERT_EQ(result.successful, 3u);

    for (const auto& key : {"d1", "d2", "d3"}) {
        auto got = f.client.get(key);
        ASSERT_TRUE(!got.has_value());
    }
    return true;
}

bool test_stats() {
    TestFixture f;
    f.client.insert("s1", {1.0f, 0.0f, 0.0f, 0.0f});
    f.client.insert("s2", {0.0f, 1.0f, 0.0f, 0.0f});
    f.client.search({1.0f, 0.0f, 0.0f, 0.0f}, 1);

    auto s = f.client.stats();
    ASSERT_TRUE(s.total_vectors >= 2);
    ASSERT_TRUE(s.total_inserts >= 2);
    ASSERT_TRUE(s.total_searches >= 1);
    return true;
}

bool test_set_algorithm() {
    TestFixture f;
    ASSERT_TRUE(f.client.set_algorithm("exact"));
    f.client.insert("alg1", {1.0f, 0.0f, 0.0f, 0.0f});
    auto results = f.client.search({1.0f, 0.0f, 0.0f, 0.0f}, 1);
    ASSERT_TRUE(results.size() >= 1);
    return true;
}

bool test_set_metric() {
    TestFixture f;
    ASSERT_TRUE(f.client.set_metric("cosine"));
    ASSERT_TRUE(f.client.set_metric("euclidean"));
    return true;
}

bool test_keep_alive() {
    // Multiple requests on the same connection
    TestFixture f;
    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(f.client.ping());
    }
    return true;
}

bool test_binary_vector_integrity() {
    // Verify float precision is preserved through the wire
    TestFixture f;
    std::vector<float> v = {3.14159265f, -2.71828f, 0.0001f, 999999.0f};
    f.client.insert("precise", v);

    auto result = f.client.get("precise");
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < v.size(); ++i) {
        ASSERT_NEAR(result->vector[i], v[i], 1e-7f);
    }
    return true;
}

bool test_concurrent_clients() {
    TestFixture f;
    // Insert some data first
    for (int i = 0; i < 20; ++i) {
        f.client.insert("cc" + std::to_string(i),
                        {static_cast<float>(i), 0.0f, 0.0f, 0.0f});
    }

    // Spawn 4 client threads doing searches concurrently
    std::atomic<int> successes{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&successes, t]() {
            TCPClient c("127.0.0.1", TEST_PORT);
            c.connect();
            for (int i = 0; i < 50; ++i) {
                auto results = c.search({static_cast<float>(t), 0.0f, 0.0f, 0.0f}, 3);
                if (!results.empty()) successes.fetch_add(1);
            }
            c.disconnect();
        });
    }

    for (auto& th : threads) th.join();
    ASSERT_EQ(successes.load(), 200);
    return true;
}

bool test_throughput() {
    TestFixture f;
    f.client.set_algorithm("exact");

    // Insert 500 vectors
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 500; ++i) {
        f.client.insert("tp" + std::to_string(i),
                        {static_cast<float>(i), static_cast<float>(i % 10),
                         static_cast<float>(i % 100), 0.0f});
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double insert_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 500 searches
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 500; ++i) {
        f.client.search({static_cast<float>(i), 0.0f, 0.0f, 0.0f}, 5);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "\n    [perf] 500 inserts: " << insert_ms << " ms ("
              << static_cast<int>(500000.0 / insert_ms) << " ops/s)\n";
    std::cout << "    [perf] 500 searches: " << search_ms << " ms ("
              << static_cast<int>(500000.0 / search_ms) << " ops/s)\n    ";

    ASSERT_TRUE(true); // always pass, just report numbers
    return true;
}

// ── Edge-case tests using raw sockets ─────────────────────────

// Open a raw TCP socket to the test server. Returns -1 on failure.
static int connect_raw() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(TEST_PORT));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd); return -1;
    }
    return fd;
}

// Server must reject a frame whose declared payload length exceeds the cap,
// then keep serving subsequent clients.
bool test_oversize_payload_rejected() {
    TestFixture f;
    int fd = connect_raw();
    ASSERT_TRUE(fd >= 0);

    // Magic + INSERT command + huge declared payload length.
    uint8_t header[proto::HEADER_SIZE];
    uint16_t magic = proto::MAGIC;
    std::memcpy(header, &magic, 2);
    header[2] = static_cast<uint8_t>(proto::Command::INSERT);
    uint32_t huge = static_cast<uint32_t>(proto::MAX_PAYLOAD_SIZE) + 1;
    std::memcpy(header + 3, &huge, 4);
    ASSERT_EQ(::send(fd, header, sizeof(header), 0), (ssize_t)sizeof(header));

    // Server should respond with an error frame, not just hang up.
    uint8_t resp[proto::HEADER_SIZE];
    ssize_t got = ::recv(fd, resp, sizeof(resp), 0);
    ASSERT_TRUE(got == (ssize_t)sizeof(resp));
    uint16_t resp_magic;
    std::memcpy(&resp_magic, resp, 2);
    ASSERT_EQ(resp_magic, proto::MAGIC);
    ASSERT_EQ(resp[2], static_cast<uint8_t>(proto::Status::ERROR));
    ::close(fd);

    // The fixture's client must still be working — server didn't crash.
    ASSERT_TRUE(f.client.ping());
    return true;
}

// Bad magic bytes → server returns an error frame; no SIGPIPE; subsequent
// clients still work.
bool test_malformed_magic_rejected() {
    TestFixture f;
    int fd = connect_raw();
    ASSERT_TRUE(fd >= 0);

    uint8_t header[proto::HEADER_SIZE]{0};
    header[0] = 0xFF; header[1] = 0xFF; // wrong magic
    header[2] = 0x00;
    uint32_t zero = 0;
    std::memcpy(header + 3, &zero, 4);
    ASSERT_EQ(::send(fd, header, sizeof(header), 0), (ssize_t)sizeof(header));

    uint8_t resp[proto::HEADER_SIZE];
    ssize_t got = ::recv(fd, resp, sizeof(resp), 0);
    ASSERT_TRUE(got == (ssize_t)sizeof(resp));
    ASSERT_EQ(resp[2], static_cast<uint8_t>(proto::Status::ERROR));
    ::close(fd);

    ASSERT_TRUE(f.client.ping());
    return true;
}

// Hostile client connects and immediately drops without sending anything.
// Server must not crash or wedge a worker.
bool test_client_abrupt_disconnect() {
    TestFixture f;
    for (int i = 0; i < 5; ++i) {
        int fd = connect_raw();
        ASSERT_TRUE(fd >= 0);
        ::close(fd);
    }
    // Allow workers to notice the EOFs.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ASSERT_TRUE(f.client.ping());

    // And a real insert/get still works after the abuse.
    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(f.client.insert("after-abuse", v, "ok"));
    auto r = f.client.get("after-abuse");
    ASSERT_TRUE(r.has_value());
    return true;
}

// Stats counters reflect what we've actually done.
bool test_stats_counters_consistent() {
    TestFixture f;
    auto before = f.client.stats();

    std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(f.client.insert("s1", v, ""));
    ASSERT_TRUE(f.client.insert("s2", v, ""));
    (void)f.client.search(v, 5);
    (void)f.client.search(v, 5);
    (void)f.client.search(v, 5);

    auto after = f.client.stats();
    ASSERT_TRUE(after.total_inserts  >= before.total_inserts  + 2);
    ASSERT_TRUE(after.total_searches >= before.total_searches + 3);
    return true;
}

// ── Main ───────────────────────────────────────────────────────

int main() {
    std::cout << "=== TCP Transport Tests ===\n\n";

    std::cout << "[Protocol basics]\n";
    RUN_TEST(test_ping);
    RUN_TEST(test_keep_alive);
    RUN_TEST(test_binary_vector_integrity);

    std::cout << "\n[CRUD operations]\n";
    RUN_TEST(test_insert_and_get);
    RUN_TEST(test_get_not_found);
    RUN_TEST(test_search);
    RUN_TEST(test_delete);
    RUN_TEST(test_delete_not_found);
    RUN_TEST(test_update);

    std::cout << "\n[Batch operations]\n";
    RUN_TEST(test_batch_insert);
    RUN_TEST(test_batch_delete);

    std::cout << "\n[Configuration]\n";
    RUN_TEST(test_set_algorithm);
    RUN_TEST(test_set_metric);
    RUN_TEST(test_stats);

    std::cout << "\n[Edge cases]\n";
    RUN_TEST(test_oversize_payload_rejected);
    RUN_TEST(test_malformed_magic_rejected);
    RUN_TEST(test_client_abrupt_disconnect);
    RUN_TEST(test_stats_counters_consistent);

    std::cout << "\n[Concurrency & performance]\n";
    RUN_TEST(test_concurrent_clients);
    RUN_TEST(test_throughput);

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run
              << " passed, " << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
