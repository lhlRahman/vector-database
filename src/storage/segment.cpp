#include "segment.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <fcntl.h>
#include <unistd.h>

#include "../utils/atomic_write.hpp"

namespace {
constexpr uint32_t kWalMagic = 0x314c5756;      // "VWL1"
constexpr uint32_t kVectorMagic = 0x31434556;   // "VEC1"
constexpr uint32_t kTombstoneMagic = 0x31424d54; // "TMB1"
constexpr uint32_t kHnswMagic = 0x31575348;     // "HSW1"
constexpr uint32_t kFormatVersion = 1;

struct WalRecordHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t sequence;
    uint32_t key_bytes;
    uint32_t metadata_bytes;
    uint32_t dimensions;
    uint32_t vector_bytes;
    uint32_t crc32;
};

// Standard CRC-32 (IEEE 802.3, reflected poly 0xEDB88320). The old checksum was
// named crc32 but was actually FNV-1a and covered ONLY the payload — a corrupt
// header (op/len/dims) could pass. This is a real CRC over header + payload.
uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

// CRC-32 over the record: every header field EXCEPT the trailing crc32, then the
// payload. (crc32 is the last real field, so offsetof gives the covered prefix.)
uint32_t wal_record_crc(const WalRecordHeader& h, const uint8_t* payload, size_t payload_len) {
    uint32_t crc = 0xFFFFFFFFu;
    crc = crc32_update(crc, reinterpret_cast<const uint8_t*>(&h), offsetof(WalRecordHeader, crc32));
    crc = crc32_update(crc, payload, payload_len);
    return ~crc;
}

template <typename T>
void write_pod(std::ostream& os, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    os.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!os.good()) throw std::runtime_error("failed to write binary value");
}

template <typename T>
bool read_pod(std::istream& is, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    is.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    return is.good();
}

void write_string(std::ostream& os, const std::string& value) {
    uint32_t size = static_cast<uint32_t>(value.size());
    write_pod(os, size);
    os.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!os.good()) throw std::runtime_error("failed to write string");
}

bool read_string(std::istream& is, std::string& value) {
    uint32_t size = 0;
    if (!read_pod(is, size)) return false;
    value.resize(size);
    if (size > 0) {
        is.read(value.data(), static_cast<std::streamsize>(size));
        if (!is.good()) return false;
    }
    return true;
}

using vdb::io::atomic_write;
using vdb::io::fsync_file;
using vdb::io::fsync_dir;

size_t file_size_or_zero(const std::filesystem::path& path) {
    std::error_code ec;
    auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<size_t>(size);
}
}

VectorSegment::VectorSegment(std::string id, std::filesystem::path directory, Config config, State state)
    : id_(std::move(id)),
      directory_(std::move(directory)),
      config_(std::move(config)),
      state_(state) {
    rebuildIndex();
}

void VectorSegment::initializeNew() {
    std::filesystem::create_directories(directory_);
    fsync_dir(directory_.parent_path());
    if (state_ == State::Mutable) {
        std::ofstream wal(walPath(), std::ios::binary | std::ios::app);
        if (!wal.is_open()) {
            throw std::runtime_error("cannot create WAL: " + walPath().string());
        }
        wal.close();
        // Persist the WAL file's directory entry. Without this, a power
        // failure between create and the first write can lose the file
        // even though create() returned successfully.
        fsync_file(walPath());
        fsync_dir(directory_);
    }
    writeMetadata();
}

void VectorSegment::load() {
    readMetadata();
    records_.clear();
    key_to_slot_.clear();
    live_count_ = 0;
    tombstone_count_ = 0;
    max_sequence_ = 0;

    if (state_ == State::Sealed) {
        readVectorsFile();
        rebuildIndex();
        if (!readHNSWSnapshot()) {
            rebuildIndex();
            for (size_t slot = 0; slot < records_.size(); ++slot) {
                if (records_[slot].active) {
                    hnsw_->insert(slot, records_[slot].key);
                }
            }
        }
        readTombstonesFile();
    } else {
        rebuildIndex();
        replayWal();
    }
}

bool VectorSegment::insert(const Vector& vector, const std::string& key, const std::string& metadata, uint64_t sequence) {
    if (state_ != State::Mutable) return false;
    if (contains(key)) return false;
    appendWalInsert(vector, key, metadata, sequence);
    return applyInsert(vector, key, metadata, sequence);
}

bool VectorSegment::update(const Vector& vector,
                           const std::string& key,
                           const std::string& metadata,
                           uint64_t sequence) {
    if (state_ != State::Mutable) return false;
    appendWalUpdate(vector, key, metadata, sequence);
    return applyUpdate(vector, key, metadata, sequence);
}

bool VectorSegment::insertRecovered(const Vector& vector,
                                    const std::string& key,
                                    const std::string& metadata,
                                    uint64_t sequence) {
    if (contains(key)) return false;
    return applyInsert(vector, key, metadata, sequence);
}

bool VectorSegment::remove(const std::string& key, uint64_t sequence) {
    if (!contains(key)) return false;

    if (state_ == State::Mutable) {
        appendWalDelete(key, sequence);
    } else {
        appendSealedTombstone(key, sequence);
    }

    bool removed = applyDelete(key, sequence);
    max_sequence_ = std::max(max_sequence_, sequence);
    return removed;
}

bool VectorSegment::contains(const std::string& key) const {
    return key_to_slot_.find(key) != key_to_slot_.end();
}

std::optional<Vector> VectorSegment::get(const std::string& key) const {
    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return std::nullopt;
    const auto& record = records_[it->second];
    return Vector(record.values);
}

std::string VectorSegment::getMetadata(const std::string& key) const {
    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return "";
    return records_[it->second].metadata;
}

std::vector<VectorSegment::SearchResult> VectorSegment::search(const Vector& query, size_t k) const {
    if (!hnsw_ || live_count_ == 0) return {};

    auto raw = hnsw_->search(query, k);
    std::vector<SearchResult> results;
    results.reserve(raw.size());

    for (const auto& [key, distance] : raw) {
        auto it = key_to_slot_.find(key);
        if (it == key_to_slot_.end()) continue;
        const auto& record = records_[it->second];
        if (!record.active) continue;
        results.push_back(SearchResult{key, distance, record.metadata, record.sequence});
    }

    return results;
}

void VectorSegment::forEachLive(const std::function<void(const RecordView&)>& visitor) const {
    for (const auto& record : records_) {
        if (!record.active) continue;
        visitor(RecordView{record.key, record.metadata, record.values.data(), record.sequence, true});
    }
}

void VectorSegment::forEachRecord(const std::function<void(const RecordView&)>& visitor) const {
    for (const auto& record : records_) {
        visitor(RecordView{
            record.key,
            record.metadata,
            record.values.data(),
            record.sequence,
            record.active,
        });
    }
}

void VectorSegment::seal() {
    if (state_ == State::Sealed) return;
    writeVectorsFile();
    writeTombstonesFile();
    writeHNSWSnapshot();
    state_ = State::Sealed;
    writeMetadata();
    // The WAL is now redundant: a sealed segment loads from vectors.bin/hnsw
    // snapshot, never the WAL. Remove it so it doesn't linger and accumulate
    // disk across seal cycles. Done last so a crash beforehand leaves only
    // harmless leftover bytes (segment.meta already records State::Sealed).
    std::error_code ec;
    std::filesystem::remove(walPath(), ec);
}

void VectorSegment::flush() {
    if (state_ == State::Sealed) {
        writeTombstonesFile();
        writeMetadata();
        return;
    }
    fsync_file(walPath());
    writeMetadata();
}

void VectorSegment::beginDeferredSync() { defer_sync_ = true; }

void VectorSegment::commitDeferredSync() {
    defer_sync_ = false;
    // One fsync for the whole batch (group commit). fsync both the WAL and, if a
    // tombstone file exists, the tombstone file.
    if (std::filesystem::exists(walPath())) fsync_file(walPath());
    if (std::filesystem::exists(tombstonesPath())) fsync_file(tombstonesPath());
}

double VectorSegment::tombstoneRatio() const {
    if (records_.empty()) return 0.0;
    return static_cast<double>(tombstone_count_) / static_cast<double>(records_.size());
}

VectorSegment::Statistics VectorSegment::getStatistics() const {
    return Statistics{
        id_,
        state_,
        records_.size(),
        live_count_,
        tombstone_count_,
        file_size_or_zero(walPath()),
        file_size_or_zero(vectorsPath()),
        file_size_or_zero(hnswPath()),
        hnsw_ ? hnsw_->getMemoryStatistics()
              : HNSWIndex::MemoryStatistics{config_.allocation_strategy, 0, 0, 0, 0, 0, 0},
    };
}

void VectorSegment::rebuildIndex() {
    auto accessor = [this](uint64_t slot_id) -> const float* {
        return vectorPtr(slot_id);
    };

    hnsw_ = std::make_unique<HNSWIndex>(
        config_.dimensions,
        config_.hnsw_m,
        config_.hnsw_ef_construction,
        config_.hnsw_ef_search,
        config_.metric,
        accessor,
        config_.allocation_strategy,
        config_.arena_initial_size);
}

bool VectorSegment::applyInsert(const Vector& vector,
                                const std::string& key,
                                const std::string& metadata,
                                uint64_t sequence) {
    if (vector.size() != config_.dimensions) {
        throw std::invalid_argument("segment vector dimension mismatch");
    }
    if (contains(key)) return false;

    Record record;
    record.key = key;
    record.metadata = metadata;
    record.values.assign(vector.begin(), vector.end());
    record.sequence = sequence;
    record.active = true;

    uint64_t slot_id = records_.size();
    records_.push_back(std::move(record));
    key_to_slot_[key] = static_cast<size_t>(slot_id);
    ++live_count_;
    max_sequence_ = std::max(max_sequence_, sequence);

    hnsw_->insert(slot_id, key);
    return true;
}

bool VectorSegment::applyUpdate(const Vector& vector,
                                const std::string& key,
                                const std::string& metadata,
                                uint64_t sequence) {
    if (vector.size() != config_.dimensions) {
        throw std::invalid_argument("segment vector dimension mismatch");
    }
    if (contains(key)) {
        applyDelete(key, sequence);
    }
    return applyInsert(vector, key, metadata, sequence);
}

bool VectorSegment::applyDelete(const std::string& key, uint64_t sequence) {
    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return false;

    size_t slot_id = it->second;
    auto& record = records_[slot_id];
    if (!record.active) {
        key_to_slot_.erase(it);
        return false;
    }

    record.active = false;
    record.sequence = std::max(record.sequence, sequence);
    key_to_slot_.erase(it);
    --live_count_;
    ++tombstone_count_;
    hnsw_->removeSlot(slot_id);
    return true;
}

void VectorSegment::appendWalInsert(const Vector& vector,
                                    const std::string& key,
                                    const std::string& metadata,
                                    uint64_t sequence) {
    std::vector<uint8_t> payload;
    payload.reserve(key.size() + metadata.size() + vector.size() * sizeof(float));
    payload.insert(payload.end(), key.begin(), key.end());
    payload.insert(payload.end(), metadata.begin(), metadata.end());
    // std::as_bytes gives a span<const std::byte>; reinterpret to uint8_t for vector::insert.
    const auto vec_bytes = std::as_bytes(std::span(vector.data_ptr(), vector.size()));
    payload.insert(payload.end(),
                   reinterpret_cast<const uint8_t*>(vec_bytes.data()),
                   reinterpret_cast<const uint8_t*>(vec_bytes.data()) + vec_bytes.size());

    WalRecordHeader header{
        kWalMagic,
        static_cast<uint16_t>(kFormatVersion),
        static_cast<uint16_t>(WalEntry::Op::Insert),
        sequence,
        static_cast<uint32_t>(key.size()),
        static_cast<uint32_t>(metadata.size()),
        static_cast<uint32_t>(config_.dimensions),
        static_cast<uint32_t>(vector.size() * sizeof(float)),
        0u,  // crc32 filled in below (covers header + payload)
    };
    header.crc32 = wal_record_crc(header, payload.data(), payload.size());

    std::ofstream os(walPath(), std::ios::binary | std::ios::app);
    if (!os.is_open()) throw std::runtime_error("cannot append WAL: " + walPath().string());
    write_pod(os, header);
    if (!payload.empty()) {
        os.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
    os.flush();
    if (!os.good()) throw std::runtime_error("failed writing WAL insert");
    // Per-record durability by default ("the call doesn't return until the WAL
    // record is on disk"); a group-commit batch defers this to one fsync in
    // commitDeferredSync().
    if (!defer_sync_) fsync_file(walPath());
}

void VectorSegment::appendWalUpdate(const Vector& vector,
                                    const std::string& key,
                                    const std::string& metadata,
                                    uint64_t sequence) {
    std::vector<uint8_t> payload;
    payload.reserve(key.size() + metadata.size() + vector.size() * sizeof(float));
    payload.insert(payload.end(), key.begin(), key.end());
    payload.insert(payload.end(), metadata.begin(), metadata.end());
    const auto vec_bytes = std::as_bytes(std::span(vector.data_ptr(), vector.size()));
    payload.insert(payload.end(),
                   reinterpret_cast<const uint8_t*>(vec_bytes.data()),
                   reinterpret_cast<const uint8_t*>(vec_bytes.data()) + vec_bytes.size());

    WalRecordHeader header{
        kWalMagic,
        static_cast<uint16_t>(kFormatVersion),
        static_cast<uint16_t>(WalEntry::Op::Update),
        sequence,
        static_cast<uint32_t>(key.size()),
        static_cast<uint32_t>(metadata.size()),
        static_cast<uint32_t>(config_.dimensions),
        static_cast<uint32_t>(vector.size() * sizeof(float)),
        0u,  // crc32 filled in below (covers header + payload)
    };
    header.crc32 = wal_record_crc(header, payload.data(), payload.size());

    std::ofstream os(walPath(), std::ios::binary | std::ios::app);
    if (!os.is_open()) throw std::runtime_error("cannot append WAL: " + walPath().string());
    write_pod(os, header);
    if (!payload.empty()) {
        os.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
    os.flush();
    if (!os.good()) throw std::runtime_error("failed writing WAL update");
    if (!defer_sync_) fsync_file(walPath());
}

void VectorSegment::appendWalDelete(const std::string& key, uint64_t sequence) {
    std::vector<uint8_t> payload(key.begin(), key.end());
    WalRecordHeader header{
        kWalMagic,
        static_cast<uint16_t>(kFormatVersion),
        static_cast<uint16_t>(WalEntry::Op::Delete),
        sequence,
        static_cast<uint32_t>(key.size()),
        0,
        static_cast<uint32_t>(config_.dimensions),
        0,
        0u,  // crc32 filled in below (covers header + payload)
    };
    header.crc32 = wal_record_crc(header, payload.data(), payload.size());

    std::ofstream os(walPath(), std::ios::binary | std::ios::app);
    if (!os.is_open()) throw std::runtime_error("cannot append WAL: " + walPath().string());
    write_pod(os, header);
    if (!payload.empty()) {
        os.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }
    os.flush();
    if (!os.good()) throw std::runtime_error("failed writing WAL delete");
    if (!defer_sync_) fsync_file(walPath());
}

void VectorSegment::appendSealedTombstone(const std::string& key, uint64_t sequence) {
    bool new_file = !std::filesystem::exists(tombstonesPath());
    std::ofstream os(tombstonesPath(), std::ios::binary | std::ios::app);
    if (!os.is_open()) throw std::runtime_error("cannot append tombstone file: " + tombstonesPath().string());

    if (new_file) {
        write_pod(os, kTombstoneMagic);
        write_pod(os, kFormatVersion);
    }

    write_pod(os, sequence);
    write_string(os, key);
    os.flush();
    if (!os.good()) throw std::runtime_error("failed writing tombstone");
    if (!defer_sync_) fsync_file(tombstonesPath());
    if (new_file) {
        // First-time create: also persist the directory entry so the
        // tombstone file isn't lost on power failure.
        fsync_dir(directory_);
    }
}

std::vector<VectorSegment::WalEntry> VectorSegment::readWal() const {
    std::vector<WalEntry> entries;
    std::ifstream is(walPath(), std::ios::binary);
    if (!is.is_open()) return entries;

    while (true) {
        WalRecordHeader header{};
        if (!read_pod(is, header)) break;
        if (header.magic != kWalMagic || header.version != kFormatVersion ||
            header.dimensions != config_.dimensions) {
            break;
        }

        size_t payload_size = static_cast<size_t>(header.key_bytes) +
                              static_cast<size_t>(header.metadata_bytes) +
                              static_cast<size_t>(header.vector_bytes);
        std::vector<uint8_t> payload(payload_size);
        if (payload_size > 0) {
            is.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload_size));
            if (!is.good()) break;
        }
        if (wal_record_crc(header, payload.data(), payload.size()) != header.crc32) break;

        WalEntry entry;
        entry.op = static_cast<WalEntry::Op>(header.op);
        entry.sequence = header.sequence;
        if (entry.op != WalEntry::Op::Insert &&
            entry.op != WalEntry::Op::Delete &&
            entry.op != WalEntry::Op::Update) {
            break;
        }

        size_t offset = 0;
        entry.key.assign(reinterpret_cast<const char*>(payload.data() + offset), header.key_bytes);
        offset += header.key_bytes;
        entry.metadata.assign(reinterpret_cast<const char*>(payload.data() + offset), header.metadata_bytes);
        offset += header.metadata_bytes;

        if (entry.op == WalEntry::Op::Insert || entry.op == WalEntry::Op::Update) {
            if (header.vector_bytes != config_.dimensions * sizeof(float)) break;
            entry.values.resize(config_.dimensions);
            std::memcpy(entry.values.data(), payload.data() + offset, header.vector_bytes);
        }

        entries.push_back(std::move(entry));
    }

    return entries;
}

void VectorSegment::replayWal() {
    for (const auto& entry : readWal()) {
        if (entry.op == WalEntry::Op::Insert) {
            Vector vector(entry.values);
            applyInsert(vector, entry.key, entry.metadata, entry.sequence);
        } else if (entry.op == WalEntry::Op::Update) {
            Vector vector(entry.values);
            applyUpdate(vector, entry.key, entry.metadata, entry.sequence);
        } else if (entry.op == WalEntry::Op::Delete) {
            applyDelete(entry.key, entry.sequence);
            max_sequence_ = std::max(max_sequence_, entry.sequence);
        }
    }
}

void VectorSegment::writeVectorsFile() const {
    atomic_write(vectorsPath(), [&](std::ostream& os) {
        write_pod(os, kVectorMagic);
        write_pod(os, kFormatVersion);
        uint64_t dimensions = config_.dimensions;
        uint64_t records = records_.size();
        write_pod(os, dimensions);
        write_pod(os, records);

        for (const auto& record : records_) {
            write_pod(os, record.sequence);
            uint8_t active = record.active ? 1 : 0;
            write_pod(os, active);
            write_string(os, record.key);
            write_string(os, record.metadata);
            os.write(reinterpret_cast<const char*>(record.values.data()),
                     static_cast<std::streamsize>(record.values.size() * sizeof(float)));
            if (!os.good()) throw std::runtime_error("failed writing vector payload");
        }
    });
}

void VectorSegment::readVectorsFile() {
    std::ifstream is(vectorsPath(), std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("missing vectors file for sealed segment: " + vectorsPath().string());
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t dimensions = 0;
    uint64_t records = 0;
    if (!read_pod(is, magic) || !read_pod(is, version) || !read_pod(is, dimensions) || !read_pod(is, records)) {
        throw std::runtime_error("corrupt vectors file header: " + vectorsPath().string());
    }
    if (magic != kVectorMagic || version != kFormatVersion || dimensions != config_.dimensions) {
        throw std::runtime_error("invalid vectors file: " + vectorsPath().string());
    }

    records_.reserve(static_cast<size_t>(records));
    for (uint64_t i = 0; i < records; ++i) {
        Record record;
        uint8_t active = 0;
        if (!read_pod(is, record.sequence) || !read_pod(is, active) ||
            !read_string(is, record.key) || !read_string(is, record.metadata)) {
            throw std::runtime_error("corrupt vector record: " + vectorsPath().string());
        }

        record.values.resize(config_.dimensions);
        is.read(reinterpret_cast<char*>(record.values.data()),
                static_cast<std::streamsize>(record.values.size() * sizeof(float)));
        if (!is.good()) throw std::runtime_error("corrupt vector payload: " + vectorsPath().string());

        record.active = active != 0;
        size_t slot = records_.size();
        records_.push_back(std::move(record));
        max_sequence_ = std::max(max_sequence_, records_.back().sequence);
        if (records_.back().active) {
            key_to_slot_[records_.back().key] = slot;
            ++live_count_;
        } else {
            ++tombstone_count_;
        }
    }
}

void VectorSegment::writeTombstonesFile() const {
    atomic_write(tombstonesPath(), [&](std::ostream& os) {
        write_pod(os, kTombstoneMagic);
        write_pod(os, kFormatVersion);
        for (const auto& record : records_) {
            if (record.active) continue;
            write_pod(os, record.sequence);
            write_string(os, record.key);
        }
    });
}

void VectorSegment::readTombstonesFile() {
    std::ifstream is(tombstonesPath(), std::ios::binary);
    if (!is.is_open()) return;

    uint32_t magic = 0;
    uint32_t version = 0;
    if (!read_pod(is, magic) || !read_pod(is, version)) return;
    if (magic != kTombstoneMagic || version != kFormatVersion) return;

    while (true) {
        uint64_t sequence = 0;
        std::string key;
        if (!read_pod(is, sequence)) break;
        if (!read_string(is, key)) break;
        applyDelete(key, sequence);
        max_sequence_ = std::max(max_sequence_, sequence);
    }
}

void VectorSegment::writeHNSWSnapshot() const {
    if (!hnsw_) return;
    auto snapshot = hnsw_->exportGraph();

    atomic_write(hnswPath(), [&](std::ostream& os) {
        write_pod(os, kHnswMagic);
        write_pod(os, kFormatVersion);
        write_pod(os, static_cast<uint64_t>(snapshot.dimensions));
        write_pod(os, static_cast<uint64_t>(snapshot.max_connections));
        write_pod(os, static_cast<uint64_t>(snapshot.max_connections_zero));
        write_pod(os, static_cast<uint64_t>(snapshot.ef_construction));
        write_pod(os, static_cast<uint64_t>(snapshot.ef_search));
        write_pod(os, static_cast<uint64_t>(snapshot.max_level));

        write_pod(os, static_cast<uint64_t>(snapshot.entry_points.size()));
        for (size_t entry : snapshot.entry_points) {
            write_pod(os, static_cast<uint64_t>(entry));
        }

        write_pod(os, static_cast<uint64_t>(snapshot.nodes.size()));
        for (const auto& node : snapshot.nodes) {
            write_pod(os, node.slot_id);
            write_pod(os, static_cast<uint64_t>(node.level));
            write_string(os, node.key);

            write_pod(os, static_cast<uint64_t>(node.neighbors.size()));
            for (size_t level = 0; level < node.neighbors.size(); ++level) {
                const auto& neighbors = node.neighbors[level];
                const auto& dists = level < node.neighbor_dists.size()
                                        ? node.neighbor_dists[level]
                                        : std::vector<float>{};
                write_pod(os, static_cast<uint64_t>(neighbors.size()));
                for (size_t neighbor : neighbors) {
                    write_pod(os, static_cast<uint64_t>(neighbor));
                }
                write_pod(os, static_cast<uint64_t>(dists.size()));
                for (float distance : dists) {
                    write_pod(os, distance);
                }
            }
        }

        write_pod(os, static_cast<uint64_t>(snapshot.deleted_keys.size()));
        for (const auto& key : snapshot.deleted_keys) {
            write_string(os, key);
        }

        write_pod(os, static_cast<uint64_t>(snapshot.deleted_slots.size()));
        for (uint64_t slot_id : snapshot.deleted_slots) {
            write_pod(os, slot_id);
        }
    });
}

bool VectorSegment::readHNSWSnapshot() {
    std::ifstream is(hnswPath(), std::ios::binary);
    if (!is.is_open()) return false;

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t dimensions = 0;
    uint64_t max_connections = 0;
    uint64_t max_connections_zero = 0;
    uint64_t ef_construction = 0;
    uint64_t ef_search = 0;
    uint64_t max_level = 0;

    if (!read_pod(is, magic) || !read_pod(is, version) ||
        !read_pod(is, dimensions) || !read_pod(is, max_connections) ||
        !read_pod(is, max_connections_zero) || !read_pod(is, ef_construction) ||
        !read_pod(is, ef_search) || !read_pod(is, max_level)) {
        return false;
    }
    if (magic != kHnswMagic || version != kFormatVersion || dimensions != config_.dimensions) return false;

    HNSWIndex::GraphSnapshot snapshot;
    snapshot.dimensions = static_cast<size_t>(dimensions);
    snapshot.max_connections = static_cast<size_t>(max_connections);
    snapshot.max_connections_zero = static_cast<size_t>(max_connections_zero);
    snapshot.ef_construction = static_cast<size_t>(ef_construction);
    snapshot.ef_search = static_cast<size_t>(ef_search);
    snapshot.max_level = static_cast<size_t>(max_level);

    uint64_t entry_count = 0;
    if (!read_pod(is, entry_count)) return false;
    snapshot.entry_points.resize(static_cast<size_t>(entry_count));
    for (uint64_t i = 0; i < entry_count; ++i) {
        uint64_t entry = 0;
        if (!read_pod(is, entry)) return false;
        snapshot.entry_points[static_cast<size_t>(i)] = static_cast<size_t>(entry);
    }

    uint64_t node_count = 0;
    if (!read_pod(is, node_count)) return false;
    snapshot.nodes.reserve(static_cast<size_t>(node_count));
    for (uint64_t i = 0; i < node_count; ++i) {
        HNSWIndex::NodeSnapshot node;
        uint64_t level = 0;
        if (!read_pod(is, node.slot_id) || !read_pod(is, level) || !read_string(is, node.key)) {
            return false;
        }
        node.level = static_cast<size_t>(level);

        uint64_t levels = 0;
        if (!read_pod(is, levels)) return false;
        node.neighbors.resize(static_cast<size_t>(levels));
        node.neighbor_dists.resize(static_cast<size_t>(levels));
        for (uint64_t level_idx = 0; level_idx < levels; ++level_idx) {
            uint64_t neighbor_count = 0;
            if (!read_pod(is, neighbor_count)) return false;
            auto& neighbors = node.neighbors[static_cast<size_t>(level_idx)];
            neighbors.resize(static_cast<size_t>(neighbor_count));
            for (uint64_t j = 0; j < neighbor_count; ++j) {
                uint64_t neighbor = 0;
                if (!read_pod(is, neighbor)) return false;
                neighbors[static_cast<size_t>(j)] = static_cast<size_t>(neighbor);
            }

            uint64_t dist_count = 0;
            if (!read_pod(is, dist_count)) return false;
            auto& dists = node.neighbor_dists[static_cast<size_t>(level_idx)];
            dists.resize(static_cast<size_t>(dist_count));
            for (uint64_t j = 0; j < dist_count; ++j) {
                if (!read_pod(is, dists[static_cast<size_t>(j)])) return false;
            }
        }
        snapshot.nodes.push_back(std::move(node));
    }

    uint64_t deleted_key_count = 0;
    if (!read_pod(is, deleted_key_count)) return false;
    snapshot.deleted_keys.resize(static_cast<size_t>(deleted_key_count));
    for (uint64_t i = 0; i < deleted_key_count; ++i) {
        if (!read_string(is, snapshot.deleted_keys[static_cast<size_t>(i)])) return false;
    }

    uint64_t deleted_slot_count = 0;
    if (!read_pod(is, deleted_slot_count)) return false;
    snapshot.deleted_slots.resize(static_cast<size_t>(deleted_slot_count));
    for (uint64_t i = 0; i < deleted_slot_count; ++i) {
        if (!read_pod(is, snapshot.deleted_slots[static_cast<size_t>(i)])) return false;
    }

    hnsw_->importGraph(snapshot);
    return true;
}

void VectorSegment::writeMetadata() const {
    atomic_write(metadataPath(), [&](std::ostream& os) {
        os << "version=1\n";
        os << "id=" << id_ << "\n";
        os << "state=" << (state_ == State::Mutable ? "mutable" : "sealed") << "\n";
        os << "dimensions=" << config_.dimensions << "\n";
        os << "records=" << records_.size() << "\n";
        os << "live_records=" << live_count_ << "\n";
        os << "tombstones=" << tombstone_count_ << "\n";
        os << "max_sequence=" << max_sequence_ << "\n";
    });
}

void VectorSegment::readMetadata() {
    std::ifstream is(metadataPath());
    if (!is.is_open()) return;

    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        if (key == "state") {
            state_ = (value == "sealed") ? State::Sealed : State::Mutable;
        } else if (key == "max_sequence") {
            max_sequence_ = std::stoull(value);
        }
    }
}

const float* VectorSegment::vectorPtr(uint64_t slot_id) const {
    if (slot_id >= records_.size()) {
        throw std::out_of_range("segment slot out of range");
    }
    return records_[static_cast<size_t>(slot_id)].values.data();
}

std::filesystem::path VectorSegment::walPath() const {
    return directory_ / "wal.log";
}

std::filesystem::path VectorSegment::vectorsPath() const {
    return directory_ / "vectors.bin";
}

std::filesystem::path VectorSegment::tombstonesPath() const {
    return directory_ / "tombstones.bin";
}

std::filesystem::path VectorSegment::hnswPath() const {
    return directory_ / "hnsw.snapshot";
}

std::filesystem::path VectorSegment::metadataPath() const {
    return directory_ / "segment.meta";
}
