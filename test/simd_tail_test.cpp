/**
 * SIMD differential tests: every kernel vs a scalar double / exact u64
 * oracle across all tail lengths (0..67) plus larger sizes, the u32
 * overflow regression, and the runtime toggle.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/core/vector.hpp"
#include "../src/optimizations/simd_operations.hpp"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    std::cerr << "FAIL: " << msg << std::endl;
    ++g_failures;
}

// magnitude = sum of |terms|, so the bound survives cancellation near zero
bool close(float actual, double expected, double magnitude,
           double rtol = 1e-5, double atol = 1e-6) {
    return std::abs(static_cast<double>(actual) - expected) <= atol + rtol * magnitude;
}

struct Oracle {
    double squared_distance = 0.0, squared_distance_mag = 0.0;
    double dot = 0.0, dot_mag = 0.0;
    double manhattan = 0.0, manhattan_mag = 0.0;
};

Oracle compute_oracle(const std::vector<float>& a, const std::vector<float>& b) {
    Oracle o;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i], y = b[i];
        const double d = x - y;
        o.squared_distance += d * d;
        o.squared_distance_mag += d * d;
        o.dot += x * y;
        o.dot_mag += std::abs(x * y);
        o.manhattan += std::abs(d);
        o.manhattan_mag += std::abs(d);
    }
    return o;
}

std::uint64_t quantized_oracle(const std::vector<std::uint8_t>& a,
                               const std::vector<std::uint8_t>& b) {
    std::uint64_t total = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        std::int64_t d = std::int64_t{a[i]} - std::int64_t{b[i]};
        total += static_cast<std::uint64_t>(d * d);
    }
    return total;
}

void test_float_kernels_at_size(size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> a(n), b(n);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    const Oracle o = compute_oracle(a, b);
    const std::string at = " at n=" + std::to_string(n);

    float sq = simd_ops::squared_distance(a.data(), b.data(), n);
    if (!close(sq, o.squared_distance, o.squared_distance_mag)) {
        fail("squared_distance" + at + ": got " + std::to_string(sq) +
             ", oracle " + std::to_string(o.squared_distance));
    }

    float dp = simd_ops::dot_product(a.data(), b.data(), n);
    if (!close(dp, o.dot, o.dot_mag)) {
        fail("dot_product" + at + ": got " + std::to_string(dp) +
             ", oracle " + std::to_string(o.dot));
    }

    float md = simd_ops::manhattan_distance(a.data(), b.data(), n);
    if (!close(md, o.manhattan, o.manhattan_mag)) {
        fail("manhattan_distance" + at + ": got " + std::to_string(md) +
             ", oracle " + std::to_string(o.manhattan));
    }

    // add/subtract are lane-wise, so exact match expected
    Vector va{std::vector<float>(a)}, vb{std::vector<float>(b)};
    Vector sum(n), diff(n);
    simd_ops::add(va, vb, sum);
    simd_ops::subtract(va, vb, diff);
    for (size_t i = 0; i < n; ++i) {
        if (sum[i] != a[i] + b[i]) { fail("add" + at + " index " + std::to_string(i)); break; }
    }
    for (size_t i = 0; i < n; ++i) {
        if (diff[i] != a[i] - b[i]) { fail("subtract" + at + " index " + std::to_string(i)); break; }
    }
}

void test_quantized_at_size(size_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<std::uint8_t> a(n), b(n);
    for (auto& x : a) x = static_cast<std::uint8_t>(dist(rng));
    for (auto& x : b) x = static_cast<std::uint8_t>(dist(rng));

    std::uint64_t expected = quantized_oracle(a, b);
    std::uint64_t got = simd_ops::quantized_l2sq_u8(a.data(), b.data(), n);
    if (got != expected) {
        fail("quantized_l2sq_u8 at n=" + std::to_string(n) + ": got " +
             std::to_string(got) + ", oracle " + std::to_string(expected));
    }
}

void test_quantized_overflow() {
    // 70000 * 255^2 > UINT32_MAX — the old u32 accumulator wrapped here
    const size_t n = 70000;
    std::vector<std::uint8_t> a(n, 255), b(n, 0);
    const std::uint64_t expected = std::uint64_t{n} * 255u * 255u;
    std::uint64_t got = simd_ops::quantized_l2sq_u8(a.data(), b.data(), n);
    if (got != expected) {
        fail("quantized_l2sq_u8 overflow: got " + std::to_string(got) +
             ", expected " + std::to_string(expected));
    }
}

void test_runtime_toggle(std::mt19937& rng) {
    // disabled SIMD must be bit-identical to sequential scalar
    const size_t n = 259;
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> a(n), b(n);
    for (auto& x : a) x = dist(rng);
    for (auto& x : b) x = dist(rng);

    float ref_sq = 0.0f, ref_dot = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        ref_sq += d * d;
        ref_dot += a[i] * b[i];
    }

    simd_ops::set_enabled(false);
    bool disabled_ok = !simd_ops::is_enabled() &&
                       simd_ops::squared_distance(a.data(), b.data(), n) == ref_sq &&
                       simd_ops::dot_product(a.data(), b.data(), n) == ref_dot;
    simd_ops::set_enabled(true);
    if (!disabled_ok) fail("set_enabled(false) did not produce exact scalar results");
    if (!simd_ops::is_enabled()) fail("set_enabled(true) did not re-enable SIMD");
}

void test_dimension_mismatch_throws() {
    Vector a(8), b(9), r(8);
    bool threw = false;
    try { (void)simd_ops::squared_distance(a, b); } catch (const std::invalid_argument&) { threw = true; }
    if (!threw) fail("squared_distance(Vector) did not throw on size mismatch");

    threw = false;
    try { simd_ops::add(a, a, r); (void)r; simd_ops::add(a, b, r); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) fail("add(Vector) did not throw on size mismatch");
}

} // namespace

int main() {
    std::mt19937 rng(20260805);

    // 0..67 covers every tail remainder of the 4/8/16/32-wide loops
    for (size_t n = 0; n <= 67; ++n) {
        test_float_kernels_at_size(n, rng);
        test_quantized_at_size(n, rng);
    }
    for (size_t n : {127, 128, 129, 255, 256, 257, 768, 1000, 1024, 1025, 4096}) {
        test_float_kernels_at_size(n, rng);
        test_quantized_at_size(n, rng);
    }

    test_quantized_overflow();
    test_runtime_toggle(rng);
    test_dimension_mismatch_throws();

    if (g_failures != 0) {
        std::cerr << g_failures << " SIMD differential test(s) failed" << std::endl;
        return 1;
    }
    std::cout << "SIMD differential tests passed (lengths 0-67, 127-4096, "
                 "overflow, toggle, mismatch)" << std::endl;
    return 0;
}
