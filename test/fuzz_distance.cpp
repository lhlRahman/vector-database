// libFuzzer-style harness for distance kernels, distance metrics, and the
// scalar quantizer.
//
// Build (example):
//   c++ -std=c++20 -fsanitize=fuzzer,address -O1 \
//       test/fuzz_distance.cpp <impl objects> -o fuzz_distance
//
// The input bytes are interpreted as two float vectors of equal length. Each
// kernel is fed the same data so their execution paths (SIMD main loop + scalar
// tail handling) are exercised across arbitrary lengths and float bit patterns
// (including NaN/Inf/denormals) that the fuzzer discovers.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>
#include <memory>
#include <exception>

#include "../src/utils/distance_metrics.hpp"
#include "../src/optimizations/simd_operations.hpp"
#include "../src/optimizations/scalar_quantization.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Two vectors of equal length are packed back-to-back in the input.
    const size_t dims = [&] {
        const size_t d = size / (2 * sizeof(float));
        return d < 256 ? d : static_cast<size_t>(256);
    }();
    if (dims == 0) {
        return 0;
    }

    // Copy out the raw floats. Use memcpy rather than reinterpreting the byte
    // pointer as float* — the input buffer is not guaranteed to be aligned for
    // float access, which would be undefined behaviour.
    std::vector<float> a(dims);
    std::vector<float> b(dims);
    std::memcpy(a.data(), data, dims * sizeof(float));
    std::memcpy(b.data(), data + dims * sizeof(float), dims * sizeof(float));

    // Prevent the optimizer from discarding the kernel calls.
    volatile float fsink = 0.0f;
    volatile uint32_t usink = 0;

    // ── Distance metrics (span-based virtual API) ─────────────
    const EuclideanDistance euclidean;
    const ManhattanDistance manhattan;
    const CosineSimilarity cosine;

    fsink += euclidean.distance_raw(std::span<const float>(a), std::span<const float>(b));
    fsink += manhattan.distance_raw(std::span<const float>(a), std::span<const float>(b));
    fsink += cosine.distance_raw(std::span<const float>(a), std::span<const float>(b));

    // ── Raw-pointer SIMD kernels ──────────────────────────────
    fsink += simd_ops::squared_distance(a.data(), b.data(), dims);
    fsink += simd_ops::dot_product(a.data(), b.data(), dims);
    fsink += simd_ops::manhattan_distance(a.data(), b.data(), dims);

    // ── Quantized integer kernel over the raw input bytes ─────
    if (size >= 2) {
        const size_t half = size / 2;
        usink += simd_ops::quantized_l2_u8(data, data + half, half);
    }

    // ── Scalar quantizer: train on the two vectors, then quantize ─
    try {
        ScalarQuantizer quantizer(dims);
        const float* vectors[2] = {a.data(), b.data()};
        quantizer.train(vectors, 2);

        std::vector<uint8_t> qa(dims);
        std::vector<uint8_t> qb(dims);
        quantizer.quantize(a.data(), qa.data());
        quantizer.quantize(b.data(), qb.data());

        usink += quantizer.distance_quantized(qa.data(), qb.data());
        fsink += quantizer.approximate_distance_sq(static_cast<uint32_t>(usink));
    } catch (const std::exception&) {
        // Swallow well-behaved exceptions; the fuzzer still flags crashes/UB.
    }

    (void)fsink;
    (void)usink;
    return 0;
}
