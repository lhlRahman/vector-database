// libFuzzer harness for VectorSegment::readWal — feeds corrupted WAL files
// at the segment recovery path. Build & run:
//   make fuzz-wal
//   ./build/fuzz_wal -runs=100000

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "../src/core/vector.hpp"
#include "../src/storage/segment.hpp"

namespace {
const std::filesystem::path& fuzz_root() {
    static const auto root = std::filesystem::temp_directory_path() /
                             ("fuzz_wal_" + std::to_string(::getpid()));
    static const bool init = [] {
        std::filesystem::create_directories(root);
        return true;
    }();
    (void)init;
    return root;
}
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Reuse a single subdirectory; libFuzzer is single-threaded by default.
    static thread_local size_t counter = 0;
    auto dir = fuzz_root() / std::to_string(counter++);
    std::filesystem::create_directories(dir);
    auto wal_path = dir / "wal.log";

    {
        std::ofstream os(wal_path, std::ios::binary | std::ios::trunc);
        if (size > 0) {
            os.write(reinterpret_cast<const char*>(data),
                     static_cast<std::streamsize>(size));
        }
    }

    try {
        VectorSegment::Config cfg;
        cfg.dimensions = 8;
        cfg.metric = std::make_shared<EuclideanDistance>();
        VectorSegment seg("fuzz-seg", dir, cfg, VectorSegment::State::Mutable);
        seg.load();  // actually parse/replay the (fuzzed) WAL — the ctor alone does not
        (void)seg.recordCount();
    } catch (const std::runtime_error&) {
    } catch (const std::invalid_argument&) {
    }

    std::filesystem::remove_all(dir);
    return 0;
}
