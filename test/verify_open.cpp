// Crash-consistency verifier: cold-open a segmented store and report its state.
// Used by the Linux dm-log-writes power-loss harness to check the DB after
// replaying the block-write log up to each flush/FUA barrier: a consistent store
// opens cleanly (WAL replay truncates at the first bad CRC) and returns a
// monotonically non-decreasing count across successive barriers.
//
//   make verify-open
//   ./build/verify_open <db_dir> [expected_min_count]
// Exit 0 = opened cleanly (and count >= expected_min if given); nonzero = failure.

#include <cstdlib>
#include <iostream>
#include <string>

#include "../src/core/vector_database.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: verify_open <db_dir> [expected_min_count]\n"; return 2; }
    std::string dir = argv[1];
    long long expected_min = (argc >= 3) ? std::atoll(argv[2]) : -1;
    try {
        VectorDatabase db(128, VectorDatabase::SearchMode::HNSW, false, true, PersistenceConfig{},
                          false, 0, dir, VectorDatabase::StorageEngine::Segmented);
        db.initialize();
        long long n = static_cast<long long>(db.vectorCount());
        db.shutdown();
        std::cout << "OK dir=" << dir << " count=" << n << "\n";
        if (expected_min >= 0 && n < expected_min) {
            std::cerr << "FAIL: count " << n << " < expected_min " << expected_min
                      << " (acknowledged writes lost)\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: open threw: " << e.what() << " (inconsistent/torn store)\n";
        return 1;
    }
}
