// libFuzzer harness for the SEALED VectorSegment load path — feeds corrupted
// HNSW snapshot files at readHNSWSnapshot() / HNSWIndex::importGraph() and
// exercises the rebuild fallback plus a post-load search. Build & run:
//   make fuzz-sealed-segment
//   ./build/fuzz_sealed_segment -runs=100000
//
// A single VALID sealed segment is built once as a template. Each fuzz input
// clones the template's vectors.bin / tombstones.bin / segment.meta and then
// overwrites hnsw.snapshot with the fuzzed bytes, so only the snapshot parser
// (with its rebuild-from-vectors fallback) sees attacker-controlled data.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "../src/core/vector.hpp"
#include "../src/storage/segment.hpp"
#include "../src/utils/distance_metrics.hpp"

namespace {

constexpr size_t kDimensions = 8;

VectorSegment::Config makeConfig() {
    VectorSegment::Config cfg;
    cfg.dimensions = kDimensions;
    cfg.metric = std::make_shared<EuclideanDistance>();
    return cfg;
}

// Build a valid sealed segment once, lazily, and return its directory. After
// seal() the directory holds vectors.bin, tombstones.bin, hnsw.snapshot and
// segment.meta.
const std::filesystem::path& templateDir() {
    static const std::filesystem::path tdir = [] {
        auto dir = std::filesystem::temp_directory_path() /
                   ("fuzz_seal_tmpl_" + std::to_string(::getpid()));
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir);

        VectorSegment tmpl("tmpl", dir, makeConfig(), VectorSegment::State::Mutable);
        tmpl.initializeNew();
        for (uint64_t i = 0; i < 6; ++i) {
            std::vector<float> values(kDimensions);
            for (size_t d = 0; d < kDimensions; ++d) {
                values[d] = static_cast<float>(i) + static_cast<float>(d) * 0.25f;
            }
            (void)tmpl.insert(Vector(std::move(values)), "k" + std::to_string(i),
                              "meta" + std::to_string(i), i + 1);
        }
        tmpl.seal();
        return dir;
    }();
    return tdir;
}

const std::filesystem::path& fuzzRoot() {
    static const std::filesystem::path root = [] {
        auto dir = std::filesystem::temp_directory_path() /
                   ("fuzz_seal_" + std::to_string(::getpid()));
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir);
        return dir;
    }();
    return root;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const auto& tmpl = templateDir();

    // Fresh per-input directory; libFuzzer is single-threaded by default.
    static size_t counter = 0;
    auto dir = fuzzRoot() / std::to_string(counter++);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir);

    // Clone the valid, non-fuzzed sidecar files from the template.
    const auto copy_opts = std::filesystem::copy_options::overwrite_existing;
    std::filesystem::copy_file(tmpl / "vectors.bin", dir / "vectors.bin", copy_opts, ec);
    std::filesystem::copy_file(tmpl / "tombstones.bin", dir / "tombstones.bin", copy_opts, ec);
    std::filesystem::copy_file(tmpl / "segment.meta", dir / "segment.meta", copy_opts, ec);

    // Overwrite the HNSW snapshot with the fuzzed bytes.
    {
        std::ofstream os(dir / "hnsw.snapshot", std::ios::binary | std::ios::trunc);
        if (size > 0) {
            os.write(reinterpret_cast<const char*>(data),
                     static_cast<std::streamsize>(size));
        }
    }

    try {
        VectorSegment seg("s", dir, makeConfig(), VectorSegment::State::Sealed);
        seg.load(); // readVectorsFile + readHNSWSnapshot(FUZZED, rebuild fallback) + readTombstonesFile
        (void)seg.recordCount();
        (void)seg.search(Vector(std::vector<float>(kDimensions, 0.5f)), 3);
    } catch (const std::exception&) {
        // Corrupt snapshots are expected to be rejected (or fall back to a
        // rebuild); only crashes / UB are interesting to the fuzzer.
    }

    std::filesystem::remove_all(dir, ec);
    return 0;
}
