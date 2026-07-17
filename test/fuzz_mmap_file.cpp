// libFuzzer harness for MMapStorage's on-disk file parser — feeds corrupted
// .vdb files at the open()/reader path. Build & run:
//   c++ -std=c++20 -fsanitize=fuzzer,address test/fuzz_mmap_file.cpp \
//       src/storage/mmap_storage.cpp -o build/fuzz_mmap_file
//   ./build/fuzz_mmap_file -runs=100000

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

#include "../src/storage/mmap_storage.hpp"

namespace {
const std::filesystem::path& fuzz_root() {
    static const auto root = std::filesystem::temp_directory_path() /
                             ("fuzz_mmap_" + std::to_string(::getpid()));
    static const bool init = [] {
        std::filesystem::create_directories(root);
        return true;
    }();
    (void)init;
    return root;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Unique per-input subdirectory; libFuzzer is single-threaded by default.
    static size_t counter = 0;
    auto dir = fuzz_root() / std::to_string(counter++);
    std::filesystem::create_directories(dir);
    auto store_path = dir / "store.vdb";

    // Write the raw fuzz bytes as the on-disk storage file.
    {
        std::ofstream os(store_path, std::ios::binary | std::ios::trunc);
        if (size > 0) {
            os.write(reinterpret_cast<const char*>(data),
                     static_cast<std::streamsize>(size));
        }
    }

    try {
        MMapStorage store(store_path.string(), /*dims=*/8);
        store.open();  // parse/validate the (fuzzed) header & mmap the file

        // Exercise the readers so ASan can catch out-of-bounds access driven by
        // attacker-controlled header fields (capacity, slot_size, dims, etc.).
        uint64_t cap = store.capacity();
        uint64_t limit = cap < 64 ? cap : 64;
        volatile float sink = 0.0f;
        for (uint64_t i = 0; i < limit; ++i) {
            bool active = store.is_active(i);
            (void)store.get_key(i);
            (void)store.get_metadata(i);
            if (active) {
                const float* p = store.vector_ptr(i);
                if (p != nullptr) {
                    for (int j = 0; j < 8; ++j) {
                        sink = sink + p[j];
                    }
                }
            }
        }
        (void)sink;
        (void)store.build_key_index();
    } catch (const std::exception&) {
        // open() legitimately throws std::runtime_error / std::system_error /
        // std::invalid_argument on corrupt / invalid / dimension-mismatch
        // files. Treat any such rejection as normal.
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return 0;
}
