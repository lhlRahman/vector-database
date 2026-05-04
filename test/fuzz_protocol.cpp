// libFuzzer harness for proto::BufferReader and the TCP server's request
// parsing. Linked against clang's libclang_rt.fuzzer.a; instrumented by gcc
// via -fsanitize-coverage=trace-pc-guard, which emits the same callbacks
// libFuzzer expects.
//
// Build & run:
//   make fuzz-protocol            # short smoke (60s)
//   ./build/fuzz_protocol -runs=1000000

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../src/api/protocol.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < proto::HEADER_SIZE) return 0;

    uint16_t magic;
    std::memcpy(&magic, data, 2);
    if (magic != proto::MAGIC) return 0;

    const auto cmd = static_cast<proto::Command>(data[2]);
    uint32_t payload_len;
    std::memcpy(&payload_len, data + 3, 4);

    if (payload_len > proto::MAX_PAYLOAD_SIZE) return 0;
    if (size < proto::HEADER_SIZE + payload_len) return 0;

    proto::BufferReader r(data + proto::HEADER_SIZE, payload_len);

    try {
        switch (cmd) {
            case proto::Command::INSERT:
            case proto::Command::UPDATE: {
                (void)r.read_string();
                const uint32_t dims = r.read_u32();
                if (dims > 4096) return 0;
                std::vector<float> tmp(dims);
                r.read_floats(tmp.data(), dims);
                (void)r.read_string();
                break;
            }
            case proto::Command::SEARCH: {
                const uint32_t dims = r.read_u32();
                if (dims > 4096) return 0;
                std::vector<float> tmp(dims);
                r.read_floats(tmp.data(), dims);
                (void)r.read_u32();
                break;
            }
            case proto::Command::DELETE:
            case proto::Command::GET:
            case proto::Command::SET_ALGORITHM:
            case proto::Command::SET_METRIC:
                (void)r.read_string();
                break;
            case proto::Command::BATCH_INSERT: {
                const uint32_t n = r.read_u32();
                if (n > 16'384) return 0;
                for (uint32_t i = 0; i < n; ++i) {
                    (void)r.read_string();
                    const uint32_t dims = r.read_u32();
                    if (dims > 4096) return 0;
                    std::vector<float> tmp(dims);
                    r.read_floats(tmp.data(), dims);
                    (void)r.read_string();
                }
                break;
            }
            case proto::Command::BATCH_DELETE: {
                const uint32_t n = r.read_u32();
                if (n > 16'384) return 0;
                for (uint32_t i = 0; i < n; ++i) (void)r.read_string();
                break;
            }
            case proto::Command::PING:
            case proto::Command::STATS:
                break;
        }
    } catch (const std::runtime_error&) {
        // expected on malformed frames
    } catch (const std::length_error&) {
        // expected on oversize-string protection
    }
    // Any other exception escapes and crashes the harness — that's the bug
    // we're hunting.
    return 0;
}
