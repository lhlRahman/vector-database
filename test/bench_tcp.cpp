#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../src/api/tcp_server.hpp"
#include "../src/api/tcp_client.hpp"
#include "../src/core/vector_database.hpp"

static const int PORT = 19091;
static const size_t DIMS = 128;

using Clock = std::chrono::high_resolution_clock;

template <typename Fn>
double bench_ms(Fn&& fn) {
    auto t0 = Clock::now();
    fn();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

std::vector<float> random_vec(size_t dims, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(dims);
    for (auto& x : v) x = d(rng);
    return v;
}

void print_row(const std::string& label, double direct_ms, double tcp_ms, int ops) {
    double direct_ops = ops * 1000.0 / direct_ms;
    double tcp_ops = ops * 1000.0 / tcp_ms;
    double overhead = ((tcp_ms / direct_ms) - 1.0) * 100.0;

    std::cout << "  " << std::left << std::setw(28) << label
              << std::right << std::setw(10) << std::fixed << std::setprecision(1) << direct_ms << " ms"
              << std::setw(12) << tcp_ms << " ms"
              << std::setw(10) << static_cast<int>(direct_ops) << "/s"
              << std::setw(10) << static_cast<int>(tcp_ops) << "/s"
              << std::setw(10) << std::setprecision(0) << overhead << "%"
              << "\n";
}

int main() {
    std::mt19937 rng(42);

    std::cout << "=== Network Transport Performance Benchmark ===\n";
    std::cout << "  Dimensions: " << DIMS << "\n";
    std::cout << "  Protocol: raw TCP + binary frames\n\n";

    // Pre-generate test data
    const int N = 1000;
    std::vector<std::vector<float>> vecs(N);
    std::vector<std::string> keys(N);
    for (int i = 0; i < N; ++i) {
        vecs[i] = random_vec(DIMS, rng);
        keys[i] = "v" + std::to_string(i);
    }
    auto query = random_vec(DIMS, rng);

    std::cout << std::left << std::setw(30) << "  Benchmark"
              << std::right << std::setw(13) << "Direct"
              << std::setw(14) << "TCP"
              << std::setw(12) << "Direct"
              << std::setw(10) << "TCP"
              << std::setw(12) << "Overhead"
              << "\n";
    std::cout << "  " << std::string(87, '-') << "\n";

    // ── Insert throughput ─────────────────────────────────

    double direct_insert_ms, tcp_insert_ms;

    {
        VectorDatabase db(DIMS, VectorDatabase::SearchMode::HNSW);
        db.initialize();
        direct_insert_ms = bench_ms([&] {
            for (int i = 0; i < N; ++i) {
                Vector v(vecs[i]);
                (void)db.insert(v, keys[i]);
            }
        });
    }

    {
        TCPServer server(DIMS, "127.0.0.1", PORT, 2);
        server.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        TCPClient client("127.0.0.1", PORT);
        client.connect();
        client.set_algorithm("hnsw");

        tcp_insert_ms = bench_ms([&] {
            for (int i = 0; i < N; ++i) {
                client.insert(keys[i], vecs[i]);
            }
        });
        client.disconnect();
        server.stop();
    }
    print_row("Insert (" + std::to_string(N) + " vectors)", direct_insert_ms, tcp_insert_ms, N);

    // ── Search latency ────────────────────────────────────

    double direct_search_ms, tcp_search_ms;
    const int SEARCHES = 1000;

    {
        VectorDatabase db(DIMS, VectorDatabase::SearchMode::HNSW);
        db.initialize();
        for (int i = 0; i < N; ++i) {
            Vector v(vecs[i]);
            (void)db.insert(v, keys[i]);
        }
        Vector q(query);
        direct_search_ms = bench_ms([&] {
            for (int i = 0; i < SEARCHES; ++i) {
                auto r = db.similaritySearch(q, 10);
                (void)r;
            }
        });
    }

    {
        TCPServer server(DIMS, "127.0.0.1", PORT, 2);
        server.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        TCPClient client("127.0.0.1", PORT);
        client.connect();
        client.set_algorithm("hnsw");
        for (int i = 0; i < N; ++i) {
            client.insert(keys[i], vecs[i]);
        }

        tcp_search_ms = bench_ms([&] {
            for (int i = 0; i < SEARCHES; ++i) {
                auto r = client.search(query, 10);
                (void)r;
            }
        });
        client.disconnect();
        server.stop();
    }
    print_row("Search top-10 (" + std::to_string(SEARCHES) + " queries)", direct_search_ms, tcp_search_ms, SEARCHES);

    // ── Get latency ───────────────────────────────────────

    double direct_get_ms, tcp_get_ms;
    const int GETS = 2000;

    {
        VectorDatabase db(DIMS, VectorDatabase::SearchMode::HNSW);
        db.initialize();
        for (int i = 0; i < N; ++i) {
            Vector v(vecs[i]);
            (void)db.insert(v, keys[i]);
        }
        direct_get_ms = bench_ms([&] {
            for (int i = 0; i < GETS; ++i) {
                auto r = db.get(keys[i % N]);
                (void)r;
            }
        });
    }

    {
        TCPServer server(DIMS, "127.0.0.1", PORT, 2);
        server.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        TCPClient client("127.0.0.1", PORT);
        client.connect();
        client.set_algorithm("hnsw");
        for (int i = 0; i < N; ++i) {
            client.insert(keys[i], vecs[i]);
        }

        tcp_get_ms = bench_ms([&] {
            for (int i = 0; i < GETS; ++i) {
                auto r = client.get(keys[i % N]);
                (void)r;
            }
        });
        client.disconnect();
        server.stop();
    }
    print_row("Get by key (" + std::to_string(GETS) + " lookups)", direct_get_ms, tcp_get_ms, GETS);

    // ── Concurrent search (4 clients) ─────────────────────

    double direct_conc_ms, tcp_conc_ms;
    const int CONC_SEARCHES = 500;
    const int NUM_CLIENTS = 4;

    {
        VectorDatabase db(DIMS, VectorDatabase::SearchMode::HNSW);
        db.initialize();
        for (int i = 0; i < N; ++i) {
            Vector v(vecs[i]);
            (void)db.insert(v, keys[i]);
        }
        Vector q(query);
        direct_conc_ms = bench_ms([&] {
            std::vector<std::thread> threads;
            for (int t = 0; t < NUM_CLIENTS; ++t) {
                threads.emplace_back([&] {
                    for (int i = 0; i < CONC_SEARCHES; ++i) {
                        auto r = db.similaritySearch(q, 10);
                        (void)r;
                    }
                });
            }
            for (auto& th : threads) th.join();
        });
    }

    {
        TCPServer server(DIMS, "127.0.0.1", PORT, 4);
        server.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(30));

        // Insert data with one client
        TCPClient setup("127.0.0.1", PORT);
        setup.connect();
        setup.set_algorithm("hnsw");
        for (int i = 0; i < N; ++i) {
            setup.insert(keys[i], vecs[i]);
        }
        setup.disconnect();

        tcp_conc_ms = bench_ms([&] {
            std::vector<std::thread> threads;
            for (int t = 0; t < NUM_CLIENTS; ++t) {
                threads.emplace_back([&] {
                    TCPClient c("127.0.0.1", PORT);
                    c.connect();
                    for (int i = 0; i < CONC_SEARCHES; ++i) {
                        auto r = c.search(query, 10);
                        (void)r;
                    }
                    c.disconnect();
                });
            }
            for (auto& th : threads) th.join();
        });
        server.stop();
    }
    print_row("Concurrent 4x" + std::to_string(CONC_SEARCHES) + " searches",
              direct_conc_ms, tcp_conc_ms, NUM_CLIENTS * CONC_SEARCHES);

    // ── Wire size comparison ──────────────────────────────

    std::cout << "\n  Wire size comparison (128-dim search request):\n";
    size_t binary_size = 7 + 4 + DIMS * 4 + 4;  // header + dims + floats + k
    std::cout << "    Binary protocol: " << binary_size << " bytes\n";
    // HTTP+JSON equivalent: POST /search HTTP/1.1\r\nContent-Type: application/json\r\n...
    // {"vector": [0.123, -0.456, ...128 floats...], "k": 10}
    // Each float ~8 chars average, plus overhead
    size_t json_size = 200 + DIMS * 9;  // ~200 bytes HTTP headers + JSON overhead
    std::cout << "    HTTP + JSON:     ~" << json_size << " bytes\n";
    std::cout << "    Reduction:       " << std::fixed << std::setprecision(1)
              << (1.0 - static_cast<double>(binary_size) / json_size) * 100.0 << "%\n";

    std::cout << "\n=== Done ===\n";
    return 0;
}
