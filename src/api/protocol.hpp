#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace proto {

// Wire format:
//   Request:  [2B magic][1B cmd][4B payload_len][payload...]
//   Response: [2B magic][1B status][4B payload_len][payload...]
//
// All multi-byte integers are little-endian (native on x86/ARM).
// Vectors are raw float arrays — no text conversion.

static constexpr uint16_t MAGIC = 0x5644; // "VD"
static constexpr size_t HEADER_SIZE = 7;  // 2 + 1 + 4

enum class Command : uint8_t {
    PING        = 0x00,
    INSERT      = 0x01,
    SEARCH      = 0x02,
    DELETE      = 0x03,
    UPDATE      = 0x04,
    GET         = 0x05,
    BATCH_INSERT = 0x06,
    BATCH_DELETE = 0x07,
    SET_ALGORITHM = 0x08,
    SET_METRIC    = 0x09,
    STATS       = 0x0A,
};

enum class Status : uint8_t {
    OK          = 0x00,
    ERROR       = 0x01,
    NOT_FOUND   = 0x02,
};

// ── Buffer writer (little-endian) ──────────────────────────────

class BufferWriter {
    std::vector<uint8_t>& buf_;
public:
    explicit BufferWriter(std::vector<uint8_t>& buf) : buf_(buf) {}

    void write_u8(uint8_t v)  { buf_.push_back(v); }
    void write_u16(uint16_t v) { auto s = buf_.size(); buf_.resize(s + 2); std::memcpy(buf_.data() + s, &v, 2); }
    void write_u32(uint32_t v) { auto s = buf_.size(); buf_.resize(s + 4); std::memcpy(buf_.data() + s, &v, 4); }
    void write_f32(float v)    { auto s = buf_.size(); buf_.resize(s + 4); std::memcpy(buf_.data() + s, &v, 4); }

    void write_string(const std::string& str) {
        write_u16(static_cast<uint16_t>(str.size()));
        auto s = buf_.size();
        buf_.resize(s + str.size());
        std::memcpy(buf_.data() + s, str.data(), str.size());
    }

    void write_floats(const float* data, size_t count) {
        auto s = buf_.size();
        buf_.resize(s + count * sizeof(float));
        std::memcpy(buf_.data() + s, data, count * sizeof(float));
    }

    void write_bytes(const uint8_t* data, size_t len) {
        auto s = buf_.size();
        buf_.resize(s + len);
        std::memcpy(buf_.data() + s, data, len);
    }
};

// ── Buffer reader (little-endian) ──────────────────────────────

class BufferReader {
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;

    void check(size_t n) const {
        if (pos_ + n > size_) throw std::runtime_error("buffer underflow");
    }
public:
    BufferReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    size_t remaining() const { return size_ - pos_; }

    uint8_t read_u8()   { check(1); uint8_t v; std::memcpy(&v, data_ + pos_, 1); pos_ += 1; return v; }
    uint16_t read_u16() { check(2); uint16_t v; std::memcpy(&v, data_ + pos_, 2); pos_ += 2; return v; }
    uint32_t read_u32() { check(4); uint32_t v; std::memcpy(&v, data_ + pos_, 4); pos_ += 4; return v; }
    float read_f32()    { check(4); float v; std::memcpy(&v, data_ + pos_, 4); pos_ += 4; return v; }

    std::string read_string() {
        uint16_t len = read_u16();
        check(len);
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    const float* read_floats_ptr(size_t count) {
        size_t bytes = count * sizeof(float);
        check(bytes);
        const float* p = reinterpret_cast<const float*>(data_ + pos_);
        pos_ += bytes;
        return p;
    }

    void read_floats(float* out, size_t count) {
        size_t bytes = count * sizeof(float);
        check(bytes);
        std::memcpy(out, data_ + pos_, bytes);
        pos_ += bytes;
    }
};

// ── Frame helpers ──────────────────────────────────────────────

inline void write_request_header(std::vector<uint8_t>& buf, Command cmd, uint32_t payload_len) {
    BufferWriter w(buf);
    w.write_u16(MAGIC);
    w.write_u8(static_cast<uint8_t>(cmd));
    w.write_u32(payload_len);
}

inline void write_response_header(std::vector<uint8_t>& buf, Status status, uint32_t payload_len) {
    BufferWriter w(buf);
    w.write_u16(MAGIC);
    w.write_u8(static_cast<uint8_t>(status));
    w.write_u32(payload_len);
}

// ── Payload layouts ────────────────────────────────────────────
//
// INSERT payload:
//   key(str) + dims(u32) + float[dims] + metadata(str)
//
// SEARCH payload:
//   dims(u32) + float[dims] + k(u32)
//
// SEARCH response:
//   count(u32) + [key(str) + distance(f32)] * count
//
// DELETE payload:
//   key(str)
//
// UPDATE payload:
//   key(str) + dims(u32) + float[dims] + metadata(str)
//
// GET response:
//   dims(u32) + float[dims] + metadata(str)
//
// BATCH_INSERT payload:
//   count(u32) + [key(str) + dims(u32) + float[dims] + metadata(str)] * count
//
// BATCH_DELETE payload:
//   count(u32) + [key(str)] * count
//
// SET_ALGORITHM / SET_METRIC payload:
//   name(str)
//
// STATS response:
//   total_vectors(u32) + total_searches(u32) + total_inserts(u32) + total_deletes(u32)
//
// PING: empty payload, OK response with empty payload
//
// ERROR response payload:
//   message(str)

static constexpr size_t MAX_PAYLOAD_SIZE = 64 * 1024 * 1024; // 64MB max

} // namespace proto
