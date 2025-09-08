// Complete atomic_persistence.cpp with WAL rotation approach

// src/features/atomic_persistence.cpp
// Copyright

#include "atomic_persistence.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <cstdio> // std::rename, std::remove


// ===========================================================
// small helpers
// ===========================================================
namespace {
inline uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}
} // namespace

// ===========================================================
// AtomicPersistence (public)
// ===========================================================

AtomicPersistence::AtomicPersistence(const PersistenceConfig& cfg)
    : config_(cfg) {
    // unify paths
    main_data_file_ = config_.data_directory + "/main.db";
}

void AtomicPersistence::initialize() {
    std::lock_guard<std::mutex> lk(mtx_);
    ensureDirectories();
    // CommitLog constructor opens the WAL file
    log_ = std::make_unique<CommitLog>(config_.log_directory,
                                       config_.log_rotation_size,
                                       config_.max_log_files);
}

void AtomicPersistence::shutdown() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (log_) {
        log_->flush();
    }
}

// durable ops (caller already mutated in-memory DB)
bool AtomicPersistence::insert(const std::string& key,
                               const Vector& v,
                               const std::string& metadata) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (recovering_.load()) return false;
    if (!log_) return false;

    log_->logInsert(key, v, metadata);
    stats_.total_logged_inserts++;
    stats_.ops_since_last_checkpoint++;
    return true;
}

bool AtomicPersistence::update(const std::string& key,
                               const Vector& v,
                               const std::string& metadata) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (recovering_.load()) return false;
    if (!log_) return false;

    log_->logUpdate(key, v, metadata);
    stats_.total_logged_updates++;
    stats_.ops_since_last_checkpoint++;
    return true;
}

bool AtomicPersistence::remove(const std::string& key) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (recovering_.load()) return false;
    if (!log_) return false;

    log_->logDelete(key);
    stats_.total_logged_deletes++;
    stats_.ops_since_last_checkpoint++;
    return true;
}

// maintenance
size_t AtomicPersistence::flush() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (log_) log_->flush();
    stats_.total_flushes++;
    // API returns a size_t; we don't track count of ops flushed here
    return 0;
}

bool AtomicPersistence::checkpoint() {
    // Marker-only (does not persist full DB without maps)
    std::lock_guard<std::mutex> lk(mtx_);
    if (!log_) return false;
    log_->logCommit();
    log_->flush();
    return true;
}

void AtomicPersistence::updateConfig(const PersistenceConfig& cfg) {
    std::lock_guard<std::mutex> lk(mtx_);
    config_ = cfg;
}

// recovery
bool AtomicPersistence::loadDatabase(std::unordered_map<std::string, Vector>& vectors,
                                     std::unordered_map<std::string, std::string>& metadata) {
    std::lock_guard<std::mutex> lk(mtx_);

    recovering_.store(true);

    // 1) load checkpoint if present
    uint64_t last_seq = 0;
    if (!loadCheckpoint(vectors, metadata, last_seq)) {
        // no snapshot yet is fine -> start empty and just replay WAL
        vectors.clear();
        metadata.clear();
        last_seq = 0;
    }

    // 2) replay WAL entries since checkpoint
    // Important: We look for entries with sequence > last_seq
    replayAll(last_seq + 1, vectors, metadata);

    recovering_.store(false);
    return true;
}

// status
AtomicPersistence::Statistics AtomicPersistence::getStatistics() const {
    std::lock_guard<std::mutex> lk(mtx_);
    Statistics s = stats_;                // copy current counters
    if (log_) {
        s.wal = log_->getStatistics();    // fill WAL stats on the copy
    }
    s.recovering = recovering_.load();    // set flag on the copy
    return s;                             // return the copy
}


// DB snapshot call (used by VectorDatabase when shouldCheckpoint() == true)
bool AtomicPersistence::saveDatabase(const std::unordered_map<std::string, Vector>& vectors,
                                     const std::unordered_map<std::string, std::string>& metadata) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!log_) return false;

    auto wal_stats = log_->getStatistics();
    // last written seq in WAL (if any)
    const uint64_t seq = (wal_stats.next_sequence > 0) ? (wal_stats.next_sequence - 1) : 0;

    std::string out_file_path;
    if (!saveCheckpointFile(vectors, metadata, seq, out_file_path)) {
        return false;
    }

    // Record checkpoint in WAL
    log_->logCheckpoint(seq, out_file_path);
    log_->flush();

    // IMPORTANT: Rotate WAL to new file (preserves sequence numbering)
    // This creates a new WAL file but continues sequence from where it left off
    log_->rotateLog();
    
    // Clean up old WAL files (keep only current one)
    cleanupOldWALFiles();
    
    last_checkpoint_wal_seq_ = seq;
    stats_.total_checkpoints++;
    
    std::cout << "[checkpoint] WAL rotated at seq " << seq 
              << ". Old WAL files cleared. New entries will start at seq " 
              << seq + 1 << std::endl;
    
    return true;
}

void AtomicPersistence::cleanupOldWALFiles() {
    try {
        std::vector<std::string> wal_files;
        for (const auto& entry : std::filesystem::directory_iterator(config_.log_directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("commit.log.") == 0) {
                    wal_files.push_back(entry.path().string());
                }
            }
        }
        
        // Sort files by name (sequence number is in the name)
        std::sort(wal_files.begin(), wal_files.end());
        
        // Keep only the current/latest WAL file, delete all others
        if (wal_files.size() > 1) {
            for (size_t i = 0; i < wal_files.size() - 1; ++i) {
                std::filesystem::remove(wal_files[i]);
                std::cout << "[checkpoint] Removed old WAL: " 
                          << std::filesystem::path(wal_files[i]).filename() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Error cleaning old WAL files: " << e.what() << std::endl;
    }
}

bool AtomicPersistence::shouldCheckpoint() const {
    // either (a) too many ops since last checkpoint OR (b) WAL too large
    const bool ops_due = stats_.ops_since_last_checkpoint >= config_.checkpoint_trigger_ops;

    bool wal_big = false;
    if (log_) {
        auto wal = log_->getStatistics();
        wal_big = (wal.current_log_size >= config_.log_rotation_size);
    }
    return ops_due || wal_big;
}

void AtomicPersistence::onCheckpointCompleted() {
    std::lock_guard<std::mutex> lk(mtx_);
    stats_.ops_since_last_checkpoint = 0;
}

// ===========================================================
// AtomicPersistence (private)
// ===========================================================

void AtomicPersistence::ensureDirectories() const {
    std::error_code ec;
    std::filesystem::create_directories(config_.data_directory, ec);
    std::filesystem::create_directories(config_.log_directory, ec);
}

void AtomicPersistence::replayAll(uint64_t since_seq,
                                  std::unordered_map<std::string, Vector>& vectors,
                                  std::unordered_map<std::string, std::string>& metadata) {
    if (!log_) return;

    // Read ALL entries from WAL (regardless of sequence)
    // Then filter for those > since_seq
    auto all_entries = log_->readAllEntries();
    
    // Filter entries with sequence >= since_seq
    std::vector<LogEntry> entries;
    for (const auto& e : all_entries) {
        if (e.sequence_number >= since_seq) {
            entries.push_back(e);
        }
    }
    
    if (entries.empty()) {
        std::cout << "[recovery] No WAL entries to replay after seq " << (since_seq - 1) << std::endl;
        auto st = log_->getStatistics();
        stats_.last_replayed_sequence = (st.next_sequence > 0 ? st.next_sequence - 1 : since_seq - 1);
        return;
    }

    std::cout << "[recovery] Found " << all_entries.size() << " total WAL entries" << std::endl;
    std::cout << "[recovery] Replaying " << entries.size() 
              << " WAL entries with seq >= " << since_seq << std::endl;

    uint64_t max_seq = since_seq ? since_seq - 1 : 0;
    int replayed_count = 0;

    for (const auto& e : entries) {
        if (!e.isValid()) continue;

        switch (e.type) {
            case LogEntryType::INSERT:
            case LogEntryType::UPDATE: {
                std::string key, meta;
                Vector vec;
                if (!decodeInsertOrUpdate(e.data, key, vec, meta)) break;
                vectors[key] = vec;
                if (meta.empty()) metadata.erase(key);
                else metadata[key] = meta;

                if (e.type == LogEntryType::INSERT) stats_.total_logged_inserts++;
                else stats_.total_logged_updates++;
                replayed_count++;
                break;
            }
            case LogEntryType::DELETE: {
                std::string key;
                if (!decodeDelete(e.data, key)) break;
                vectors.erase(key);
                metadata.erase(key);
                stats_.total_logged_deletes++;
                replayed_count++;
                break;
            }
            case LogEntryType::CHECKPOINT:
                // Skip checkpoint entries during replay
                std::cout << "[recovery] Skipping checkpoint entry at seq " << e.sequence_number << std::endl;
                break;
            case LogEntryType::COMMIT:
            default:
                break;
        }
        max_seq = std::max(max_seq, e.sequence_number);
    }

    stats_.last_replayed_sequence = max_seq;
    std::cout << "[recovery] Replay complete. Replayed " << replayed_count 
              << " operations. Last sequence: " << max_seq << std::endl;
}

bool AtomicPersistence::loadCheckpoint(std::unordered_map<std::string, Vector>& vectors,
                                       std::unordered_map<std::string, std::string>& metadata,
                                       uint64_t& out_seq) {
    const std::string& path = main_data_file_;

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        out_seq = 0;
        std::cout << "[recovery] No checkpoint file found at " << path << std::endl;
        return false; // no snapshot yet
    }

    auto read_u32 = [](std::ifstream& f, uint32_t& v)->bool {
        return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof(v)));
    };
    auto read_u64 = [](std::ifstream& f, uint64_t& v)->bool {
        return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof(v)));
    };
    auto read_exact = [](std::ifstream& f, void* p, size_t n)->bool {
        return static_cast<bool>(f.read(reinterpret_cast<char*>(p), n));
    };

    uint32_t magic=0, version=0;
    if (!read_u32(in, magic) || !read_u32(in, version)) return false;
    if (magic != 0x56444244 || version != 1) return false; // "VDBD", v1

    uint64_t seq=0, ts_us=0, count=0;
    if (!read_u64(in, seq) || !read_u64(in, ts_us) || !read_u64(in, count)) return false;

    std::cout << "[recovery] Loading checkpoint with " << count 
              << " vectors at seq " << seq << std::endl;

    std::unordered_map<std::string, Vector> tmp_vectors;
    std::unordered_map<std::string, std::string> tmp_meta;
    tmp_vectors.reserve(static_cast<size_t>(count));
    tmp_meta.reserve(static_cast<size_t>(count));

    uint32_t footer_crc = 0;

    for (uint64_t i = 0; i < count; ++i) {
        // key
        uint32_t key_len = 0;
        if (!read_u32(in, key_len)) return false;
        std::string key(key_len, '\0');
        if (key_len && !read_exact(in, key.data(), key_len)) return false;
        footer_crc ^= key_len;

        // vector
        uint32_t dims = 0;
        if (!read_u32(in, dims)) return false;
        std::vector<float> buf(dims);
        if (dims && !read_exact(in, buf.data(), sizeof(float) * dims)) return false;
        footer_crc ^= dims;
        Vector vec(std::move(buf));

        // metadata
        uint32_t meta_len = 0;
        if (!read_u32(in, meta_len)) return false;
        std::string meta(meta_len, '\0');
        if (meta_len && !read_exact(in, meta.data(), meta_len)) return false;
        footer_crc ^= meta_len;

        tmp_vectors.emplace(key, std::move(vec));
        if (meta_len) tmp_meta[key] = std::move(meta);
    }

    // footer
    uint32_t footer_magic=0, crc_read=0;
    if (!read_u32(in, footer_magic) || !read_u32(in, crc_read)) return false;
    if (footer_magic != 0x454E444D) return false;      // 'ENDM'
    if (crc_read != footer_crc) return false;          // simple checksum

    vectors.swap(tmp_vectors);
    metadata.swap(tmp_meta);
    out_seq = seq;
    
    std::cout << "[recovery] Checkpoint loaded successfully. Sequence: " << seq << std::endl;
    return true;
}

bool AtomicPersistence::saveCheckpointFile(
    const std::unordered_map<std::string, Vector>& vectors,
    const std::unordered_map<std::string, std::string>& metadata,
    uint64_t sequence,
    std::string& out_file)
{
    // temp filename in data/
    const std::string tmp        = config_.data_directory + "/checkpoint_" + std::to_string(sequence) + ".tmp";
    const std::string final_path = config_.data_directory + "/main.db";

    try {
        AtomicFileWriter writer(tmp);

        // ---------- Header ----------
        // magic "VDBD" (0x56444244), version 1, sequence, timestamp (us)
        writer.write<uint32_t>(0x56444244);
        writer.write<uint32_t>(1);
        writer.write<uint64_t>(sequence);

        const auto now_us_val = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count()
        );
        writer.write<uint64_t>(now_us_val);

        // ---------- Payload ----------
        // number of vectors
        writer.write<uint64_t>(static_cast<uint64_t>(vectors.size()));

        // compute footer checksum
        uint32_t footer_crc = 0;

        for (const auto& kv : vectors) {
            const std::string& key = kv.first;
            const Vector&      vec = kv.second;

            // key
            const uint32_t key_len = static_cast<uint32_t>(key.size());
            writer.write<uint32_t>(key_len);
            if (key_len) writer.write(key.data(), key_len);
            footer_crc ^= key_len;

            // vector
            const uint32_t dims = static_cast<uint32_t>(vec.size());
            writer.write<uint32_t>(dims);
            if (dims) writer.write(vec.data_ptr(), dims * sizeof(float));
            footer_crc ^= dims;

            // metadata (optional)
            auto it = metadata.find(key);
            const std::string meta = (it != metadata.end()) ? it->second : std::string{};
            const uint32_t meta_len = static_cast<uint32_t>(meta.size());
            writer.write<uint32_t>(meta_len);
            if (meta_len) writer.write(meta.data(), meta_len);
            footer_crc ^= meta_len;
        }

        // ---------- Footer ----------
        writer.write<uint32_t>(0x454E444D);   // "ENDM"
        writer.write<uint32_t>(footer_crc);

        // atomically flush to tmp
        writer.commit();

        // move tmp -> main.db
        if (std::rename(tmp.c_str(), final_path.c_str()) != 0) {
            // if rename fails, try to clean tmp and error out
            std::remove(tmp.c_str());
            std::cerr << "saveCheckpointFile: rename(" << tmp << " -> " << final_path << ") failed\n";
            return false;
        }

        out_file = final_path;

        std::cout << "[checkpoint] wrote " << vectors.size()
                  << " vectors at seq " << sequence
                  << " -> " << final_path << std::endl;

        return true;
    } catch (const std::exception& e) {
        // best effort cleanup
        std::remove(tmp.c_str());
        std::cerr << "saveCheckpointFile failed: " << e.what() << std::endl;
        return false;
    }
}

// ===========================================================
// decode helpers (match commit_log.cpp inline-string payloads)
// ===========================================================
static bool read_u32_vec(const std::vector<uint8_t>& b, size_t& off, uint32_t& out) {
    if (off + sizeof(uint32_t) > b.size()) return false;
    std::memcpy(&out, b.data() + off, sizeof(uint32_t));
    off += sizeof(uint32_t);
    return true;
}
static bool read_bytes_vec(const std::vector<uint8_t>& b, size_t& off, void* dst, size_t n) {
    if (off + n > b.size()) return false;
    std::memcpy(dst, b.data() + off, n);
    off += n;
    return true;
}
static bool read_string_vec(const std::vector<uint8_t>& b, size_t& off, std::string& out) {
    uint32_t n = 0;
    if (!read_u32_vec(b, off, n)) return false;
    if (off + n > b.size()) return false;
    out.assign(reinterpret_cast<const char*>(b.data() + off), n);
    off += n;
    return true;
}

bool AtomicPersistence::decodeInsertOrUpdate(const std::vector<uint8_t>& blob,
                                             std::string& out_key,
                                             Vector& out_vec,
                                             std::string& out_meta) {
    size_t off = 0;

    // key
    if (!read_string_vec(blob, off, out_key)) return false;

    // vector
    uint32_t dims = 0;
    if (!read_u32_vec(blob, off, dims)) return false;
    if (off + dims * sizeof(float) > blob.size()) return false;
    std::vector<float> tmp(dims);
    if (dims && !read_bytes_vec(blob, off, tmp.data(), sizeof(float) * dims)) return false;
    out_vec = Vector(std::move(tmp));

    // metadata
    if (!read_string_vec(blob, off, out_meta)) return false;

    return true;
}

bool AtomicPersistence::decodeDelete(const std::vector<uint8_t>& blob, std::string& out_key) {
    size_t off = 0;
    return read_string_vec(blob, off, out_key);
}