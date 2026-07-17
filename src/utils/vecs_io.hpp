#pragma once

// Readers/writers for the TEXMEX .fvecs/.ivecs/.bvecs formats used by the
// standard ANN benchmarks (SIFT1M, GIST1M, ...). Each record is:
//     [int32 dim][dim * element]           (all little-endian)
// with `dim` repeated on every row. Elements are float32 (.fvecs),
// int32 (.ivecs, ground-truth neighbor ids), or uint8 (.bvecs).

#include <bit>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vecs_io {

static_assert(std::endian::native == std::endian::little,
              "vecs_io assumes a little-endian host (matches the TEXMEX format)");

struct Matrix {
    std::vector<float> data;  // row-major, n*d
    size_t n = 0;
    size_t d = 0;
    const float* row(size_t i) const { return data.data() + i * d; }
};

struct IntMatrix {
    std::vector<int32_t> data;  // row-major, n*d
    size_t n = 0;
    size_t d = 0;
    const int32_t* row(size_t i) const { return data.data() + i * d; }
};

// Generic reader: parses [int32 dim][dim*ElemT] records, converting each element
// to float. The 4-byte dim prefix on every row is consumed and must be constant.
template <typename ElemT>
inline Matrix load_vecs_as_float(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("vecs_io: cannot open " + path);

    Matrix m;
    int32_t d0 = -1;
    std::vector<ElemT> row;
    for (;;) {
        int32_t d;
        if (!in.read(reinterpret_cast<char*>(&d), 4)) break;  // clean EOF
        if (d <= 0 || d > (1 << 24)) throw std::runtime_error("vecs_io: bad dim in " + path);
        if (d0 < 0) {
            d0 = d;
            m.d = static_cast<size_t>(d);
        } else if (d != d0) {
            throw std::runtime_error("vecs_io: ragged dimensions in " + path);
        }
        row.resize(static_cast<size_t>(d));
        if (!in.read(reinterpret_cast<char*>(row.data()), sizeof(ElemT) * static_cast<size_t>(d)))
            throw std::runtime_error("vecs_io: truncated record in " + path);
        for (int32_t i = 0; i < d; ++i) m.data.push_back(static_cast<float>(row[static_cast<size_t>(i)]));
        ++m.n;
    }
    return m;
}

inline Matrix load_fvecs(const std::string& path) { return load_vecs_as_float<float>(path); }
inline Matrix load_bvecs(const std::string& path) { return load_vecs_as_float<uint8_t>(path); }

// .ivecs: rows of int32 (e.g. ground-truth neighbor ids), kept as int32.
inline IntMatrix load_ivecs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("vecs_io: cannot open " + path);

    IntMatrix m;
    int32_t d0 = -1;
    for (;;) {
        int32_t d;
        if (!in.read(reinterpret_cast<char*>(&d), 4)) break;
        if (d <= 0 || d > (1 << 24)) throw std::runtime_error("vecs_io: bad dim in " + path);
        if (d0 < 0) {
            d0 = d;
            m.d = static_cast<size_t>(d);
        } else if (d != d0) {
            throw std::runtime_error("vecs_io: ragged dimensions in " + path);
        }
        size_t base = m.data.size();
        m.data.resize(base + static_cast<size_t>(d));
        if (!in.read(reinterpret_cast<char*>(m.data.data() + base), 4 * static_cast<size_t>(d)))
            throw std::runtime_error("vecs_io: truncated record in " + path);
        ++m.n;
    }
    return m;
}

// Write a row-major float buffer as .fvecs (used to materialize synthetic data
// so the same loader path is exercised on synthetic and real datasets).
inline void write_fvecs(const std::string& path, const std::vector<float>& data, size_t n, size_t d) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("vecs_io: cannot write " + path);
    const int32_t dd = static_cast<int32_t>(d);
    for (size_t i = 0; i < n; ++i) {
        out.write(reinterpret_cast<const char*>(&dd), 4);
        out.write(reinterpret_cast<const char*>(data.data() + i * d),
                  static_cast<std::streamsize>(4 * d));
    }
}

}  // namespace vecs_io
