// libFuzzer INTEGRATION harness: drives a real VectorDatabase (default
// Segmented engine) through a fuzzed sequence of operations.
//
// Build (example):
//   c++ -std=c++20 -g -O1 -fsanitize=fuzzer,address \
//       test/fuzz_db_ops.cpp <db + deps>.cpp -o fuzz_db_ops
//
// There is no main(); libFuzzer supplies it. Each input byte-stream is walked
// as an opcode stream that mutates one long-lived, file-static database.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

#include "../src/core/vector.hpp"
#include "../src/core/vector_database.hpp"
#include "../src/utils/distance_metrics.hpp"

namespace {

// One long-lived database, constructed lazily on first use. Rebuilding it per
// input would be far too slow: segmented inserts fsync, so we amortize a single
// engine across every fuzz iteration and let its destructor clean the temp path
// at process exit.
VectorDatabase& db() {
    static VectorDatabase* instance = [] {
        namespace fs = std::filesystem;
        const fs::path storage_path =
            fs::temp_directory_path() / ("fuzz_dbops_" + std::to_string(::getpid()));
        std::error_code ec;
        fs::create_directories(storage_path, ec);  // best-effort; engine may also create it

        auto* d = new VectorDatabase(
            /*dimensions=*/8,
            VectorDatabase::SearchMode::HNSW,
            /*enable_atomic_persistence=*/false,
            /*enable_batch_operations=*/true,
            /*persistence_config=*/{},
            /*enable_query_cache=*/true,
            /*cache_capacity=*/128,
            /*storage_path=*/storage_path.string(),
            VectorDatabase::StorageEngine::Segmented);
        d->initialize();
        return d;
    }();
    return *instance;
}

// Cursor over the fuzz input. Reads past the end yield zeros so that decoding
// never needs a bounds check at every call site.
struct ByteStream {
    const uint8_t* data;
    size_t size;
    size_t pos = 0;

    uint8_t next() { return pos < size ? data[pos++] : uint8_t{0}; }
};

// Build a dim-8 vector from the next 8 bytes; missing bytes decode to 0.
Vector makeVector(ByteStream& in) {
    std::vector<float> values(8);
    for (int i = 0; i < 8; ++i) {
        values[static_cast<size_t>(i)] = static_cast<float>(in.next()) / 255.0f;
    }
    return Vector(std::move(values));
}

// Keys drawn from a small fixed pool so ops actually collide / hit.
std::string makeKey(uint8_t b) {
    return "k" + std::to_string(b % 32);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    ByteStream in{data, size};
    VectorDatabase& database = db();

    constexpr int kMaxOps = 64;
    for (int op = 0; op < kMaxOps && in.pos < in.size; ++op) {
        const uint8_t opcode = in.next();

        // Every op is wrapped so that expected validation failures (dim
        // mismatch, missing keys, etc.) surface as exceptions instead of
        // aborting the run and hiding real crashes.
        try {
            switch (opcode % 8) {
                case 0: {  // insert
                    const std::string key = makeKey(in.next());
                    Vector v = makeVector(in);
                    (void)database.insert(v, key);
                    break;
                }
                case 1: {  // update
                    const std::string key = makeKey(in.next());
                    Vector v = makeVector(in);
                    (void)database.update(v, key, "");
                    break;
                }
                case 2: {  // remove
                    const std::string key = makeKey(in.next());
                    (void)database.remove(key);
                    break;
                }
                case 3: {  // get
                    const std::string key = makeKey(in.next());
                    std::optional<Vector> got = database.get(key);
                    (void)got;
                    break;
                }
                case 4: {  // similaritySearch
                    const size_t k = static_cast<size_t>(in.next() % 16) + 1;
                    Vector q = makeVector(in);
                    auto results = database.similaritySearch(q, k);
                    (void)results;
                    break;
                }
                case 5: {  // setSearchMode
                    const uint8_t b = in.next();
                    database.setSearchMode((b & 1) ? VectorDatabase::SearchMode::HNSW
                                                   : VectorDatabase::SearchMode::Exact);
                    break;
                }
                case 6: {  // setDistanceMetric
                    const uint8_t b = in.next();
                    std::shared_ptr<DistanceMetric> metric;
                    switch (b % 3) {
                        case 0:
                            metric = std::make_shared<EuclideanDistance>();
                            break;
                        case 1:
                            metric = std::make_shared<ManhattanDistance>();
                            break;
                        default:
                            metric = std::make_shared<CosineSimilarity>();
                            break;
                    }
                    database.setDistanceMetric(std::move(metric));
                    break;
                }
                case 7: {  // insert with metadata
                    const std::string key = makeKey(in.next());
                    Vector v = makeVector(in);
                    const std::string metadata = "m" + std::to_string(in.next());
                    (void)database.insert(v, key, metadata);
                    break;
                }
            }
        } catch (const std::exception&) {
            // Expected validation path — keep fuzzing.
        }
    }

    return 0;
}
