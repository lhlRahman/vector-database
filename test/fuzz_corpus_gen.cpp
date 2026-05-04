// Tiny seed-corpus generator for the protocol fuzzer. Writes one file per
// command type into `test/corpus/protocol/`. libFuzzer mutates from these
// instead of having to discover the magic bytes from random input.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "../src/api/protocol.hpp"

namespace {
void emit(const std::filesystem::path& dir, const std::string& name,
          const std::vector<uint8_t>& bytes) {
    std::ofstream os(dir / name, std::ios::binary);
    os.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
}

std::vector<uint8_t> frame(proto::Command cmd, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> out;
    proto::write_request_header(out, cmd, static_cast<uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}
} // namespace

int main(int, char**) {
    auto dir = std::filesystem::path("test/corpus/protocol");
    std::filesystem::create_directories(dir);

    {
        std::vector<uint8_t> p; proto::BufferWriter w(p);
        w.write_string("hello");
        w.write_u32(3);
        float v[]{1.0f, 2.0f, 3.0f};
        w.write_floats(v, 3);
        w.write_string("meta");
        emit(dir, "insert", frame(proto::Command::INSERT, p));
        emit(dir, "update", frame(proto::Command::UPDATE, p));
    }
    {
        std::vector<uint8_t> p; proto::BufferWriter w(p);
        w.write_u32(3);
        float v[]{0.1f, 0.2f, 0.3f};
        w.write_floats(v, 3);
        w.write_u32(5);
        emit(dir, "search", frame(proto::Command::SEARCH, p));
    }
    {
        std::vector<uint8_t> p; proto::BufferWriter w(p);
        w.write_string("k");
        emit(dir, "delete", frame(proto::Command::DELETE, p));
        emit(dir, "get",    frame(proto::Command::GET, p));
        emit(dir, "set_alg",frame(proto::Command::SET_ALGORITHM, p));
        emit(dir, "set_met",frame(proto::Command::SET_METRIC, p));
    }
    {
        std::vector<uint8_t> p; proto::BufferWriter w(p);
        w.write_u32(2);
        for (int i = 0; i < 2; ++i) {
            w.write_string("k" + std::to_string(i));
            w.write_u32(2);
            float v[]{float(i), float(i + 1)};
            w.write_floats(v, 2);
            w.write_string("m");
        }
        emit(dir, "batch_insert", frame(proto::Command::BATCH_INSERT, p));
    }
    {
        std::vector<uint8_t> p; proto::BufferWriter w(p);
        w.write_u32(2);
        w.write_string("k0");
        w.write_string("k1");
        emit(dir, "batch_delete", frame(proto::Command::BATCH_DELETE, p));
    }
    emit(dir, "ping",  frame(proto::Command::PING,  {}));
    emit(dir, "stats", frame(proto::Command::STATS, {}));

    std::printf("Wrote %zu seeds to %s\n", std::filesystem::directory_iterator(dir) ==
                std::filesystem::directory_iterator() ? size_t{0} : size_t{11},
                dir.string().c_str());
    return 0;
}
