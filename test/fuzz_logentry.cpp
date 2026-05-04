// libFuzzer harness for LogEntry::deserialize. The WAL persists these as
// length-prefixed records on disk; corruption that gets past `isValid()`
// would mean the FNV-1a checksum is collision-prone, and corruption that
// crashes the parser is straight-up RCE bait.

#include <cstdint>
#include <vector>

#include "../src/features/commit_log.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::vector<uint8_t> buf(data, data + size);
    try {
        LogEntry e = LogEntry::deserialize(buf);
        // Serialize and round-trip — exercises calculateChecksum on
        // arbitrary payloads.
        if (e.isValid()) {
            auto reser = e.serialize();
            (void)LogEntry::deserialize(reser);
        }
    } catch (const std::length_error&) {
        // expected: oversize payload guard
    } catch (const std::bad_alloc&) {
        // accepted: an attacker-supplied length field claimed too much
    }
    return 0;
}
