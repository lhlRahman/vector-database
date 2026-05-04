// Fallback main() for fuzz harnesses when libFuzzer isn't available.
//
// Drives `LLVMFuzzerTestOneInput` with a random + mutation loop. Same crash
// detection as libFuzzer (everything runs under ASan/UBSan), minus the
// coverage-guided mutation. Install clang and rebuild with -fsanitize=fuzzer
// to get real libFuzzer.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

namespace {

void mutate(std::vector<uint8_t>& b, std::mt19937& rng) {
    if (b.empty()) return;
    std::uniform_int_distribution<size_t> pos(0, b.size() - 1);
    const size_t p = pos(rng);
    switch (rng() % 5) {
        case 0: b[p] ^= 1u << (rng() % 8); break;
        case 1: b[p] = static_cast<uint8_t>(rng()); break;
        case 2: b[p] = 0; break;
        case 3: b[p] = 0xFF; break;
        case 4: { // 4-byte length-style overwrite, safely clamped
            if (b.size() < 4) break;
            const size_t base = std::min(p & ~size_t{3}, b.size() - 4);
            const uint32_t v = (rng() % 4 == 0) ? 0xFFFF'FFFFu
                                                : static_cast<uint32_t>(rng());
            std::memcpy(b.data() + base, &v, 4);
            break;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    size_t max_runs = 200'000;
    int max_seconds = 60;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("-runs=", 0) == 0)            max_runs    = std::stoull(a.substr(6));
        else if (a.rfind("-max_total_time=", 0) == 0)
                                                  max_seconds = std::stoi(a.substr(16));
    }

    std::printf("[fuzz fallback] runs<=%zu time<=%ds — install clang for real libFuzzer\n",
                max_runs, max_seconds);

    std::mt19937 rng(0xC0FFEEu);
    std::vector<uint8_t> buf;
    buf.reserve(1024);

    auto t0 = std::chrono::steady_clock::now();
    size_t runs = 0;
    for (; runs < max_runs; ++runs) {
        // Mostly small inputs, occasionally larger.
        const size_t target = (rng() % 32 == 0) ? (rng() % 4096) : (rng() % 256);
        buf.resize(target);
        for (auto& byte : buf) byte = static_cast<uint8_t>(rng());
        // Half the time, do extra mutations (simulates structured fuzzing).
        if (rng() & 1) {
            const int rounds = static_cast<int>(rng() % 8);
            for (int r = 0; r < rounds; ++r) mutate(buf, rng);
        }
        LLVMFuzzerTestOneInput(buf.data(), buf.size());

        if ((runs & 0xFFFF) == 0) {
            auto dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            if (dt > max_seconds) break;
        }
    }
    auto dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[fuzz fallback] %zu runs in %.2fs (%.0f exec/s)\n",
                runs, dt, runs / std::max(dt, 1e-9));
    return 0;
}
