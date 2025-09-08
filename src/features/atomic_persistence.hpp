// Copyright
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../core/vector.hpp"
#include "atomic_file_writer.hpp"
#include "commit_log.hpp"
#include "recovery_state_machine.hpp"
#include "../json.hpp"  // local single-file nlohmann json

using json = nlohmann::json;

// ========================== PersistenceConfig ==========================
struct PersistenceConfig {
    // Write-Ahead Log (WAL)
    std::string log_directory      = "logs";
    size_t      log_rotation_size  = 100 * 1024 * 1024; // 100MB
    size_t      max_log_files      = 10;

    // Checkpointing (main snapshot file lives in data/main.db)
    std::string data_directory     = "data";
    std::chrono::minutes checkpoint_interval{60};
    uint64_t    checkpoint_trigger_ops = 10'000; // default to 10k

    // Misc toggles
    bool auto_recovery      = true;
    bool validate_checksums = true;
    bool enable_compression = false; // reserved
    bool enable_async_flush = true;  // reserved
};

inline void to_json(json& j, const PersistenceConfig& c) {
    j = json{
        {"log_directory", c.log_directory},
        {"log_rotation_size", c.log_rotation_size},
        {"max_log_files", c.max_log_files},
        {"data_directory", c.data_directory},
        {"checkpoint_interval_ms", c.checkpoint_interval.count()},
        {"checkpoint_trigger_ops", c.checkpoint_trigger_ops},
        {"auto_recovery", c.auto_recovery},
        {"validate_checksums", c.validate_checksums},
        {"enable_compression", c.enable_compression},
        {"enable_async_flush", c.enable_async_flush}
    };
}

// ========================== AtomicPersistence ==========================
class AtomicPersistence {
public:
    struct Statistics {
        uint64_t total_logged_inserts{0};
        uint64_t total_logged_updates{0};
        uint64_t total_logged_deletes{0};
        uint64_t total_checkpoints{0};
        uint64_t total_flushes{0};
        uint64_t last_replayed_sequence{0};
        uint64_t ops_since_last_checkpoint{0};
        bool     recovering{false};
        CommitLog::Statistics wal{};
    };

    explicit AtomicPersistence(const PersistenceConfig& cfg = PersistenceConfig{});
    ~AtomicPersistence() = default;

    // lifecycle
    void initialize();   // create dirs, open WAL, recover, start janitor thread
    void shutdown();     // flush WAL, stop janitor thread

    // durable ops (caller already mutated in-memory DB)
    bool insert(const std::string& key, const Vector& v, const std::string& metadata);
    bool update(const std::string& key, const Vector& v, const std::string& metadata);
    bool remove(const std::string& key);

    // convenience cache hook (optional, no-op here)
    std::pair<Vector*, std::string*> get(const std::string& /*key*/) { return {nullptr, nullptr}; }

    // maintenance
    size_t flush();                 // fsync WAL
    bool   checkpoint();            // marker-only; real snapshot done via saveDatabase(...)
    void   updateConfig(const PersistenceConfig& cfg);

    // recovery: load DB from checkpoint + WAL
    bool loadDatabase(std::unordered_map<std::string, Vector>& vectors,
                      std::unordered_map<std::string, std::string>& metadata);

    // DB-coordination for auto-checkpointing
    bool shouldCheckpoint() const;  // check thresholds (ops or WAL size or time)
    void onCheckpointCompleted();   // reset counters after DB saves snapshot

    // snapshot I/O (DB calls this to persist full state to main.db)
    bool saveDatabase(const std::unordered_map<std::string, Vector>& vectors,
                      const std::unordered_map<std::string, std::string>& metadata);

    // status
    Statistics getStatistics() const;
    RecoveryStateMachine::RecoveryInfo getRecoveryInfo() const { return recovery_info_; }
    bool isRecovering() const { return recovering_.load(); }

private:
    // helpers
    void ensureDirectories() const;
    void replayAll(uint64_t since_seq,
                   std::unordered_map<std::string, Vector>& vectors,
                   std::unordered_map<std::string, std::string>& metadata);

    // checkpoint I/O (simple binary format)
    bool loadCheckpoint(std::unordered_map<std::string, Vector>& vectors,
                        std::unordered_map<std::string, std::string>& metadata,
                        uint64_t& out_seq);
    bool saveCheckpointFile(const std::unordered_map<std::string, Vector>& vectors,
                            const std::unordered_map<std::string, std::string>& metadata,
                            uint64_t sequence,
                            std::string& out_file);

    // decode helpers for inline-string WAL payloads (matches commit_log.cpp)
    static bool decodeInsertOrUpdate(const std::vector<uint8_t>& blob,
                                     std::string& out_key,
                                     Vector& out_vec,
                                     std::string& out_meta);
    static bool decodeDelete(const std::vector<uint8_t>& blob, std::string& out_key);

    // Clean up old WAL files after checkpoint
    void cleanupOldWALFiles();

    // internal soft trigger (no DB maps available)
    void maybeCheckpoint_NoMaps();

private:
    PersistenceConfig                 config_;
    std::unique_ptr<CommitLog>        log_;
    mutable std::mutex                mtx_;

    // state / stats
    std::atomic<bool>                 recovering_{false};
    Statistics                        stats_{};
    RecoveryStateMachine::RecoveryInfo recovery_info_{};

    // counters for auto-checkpoint
    uint64_t                          last_checkpoint_wal_seq_{0};

    // checkpoint file
    std::string                       main_data_file_; // data/main.db

    // janitor thread for periodic checks
    std::thread                       janitor_;
    std::atomic<bool>                 janitor_stop_{true};
};

// pretty stats JSON
inline void to_json(json& j, const AtomicPersistence::Statistics& s) {
    j = json{
        {"total_logged_inserts", s.total_logged_inserts},
        {"total_logged_updates", s.total_logged_updates},
        {"total_logged_deletes", s.total_logged_deletes},
        {"total_checkpoints", s.total_checkpoints},
        {"total_flushes", s.total_flushes},
        {"last_replayed_sequence", s.last_replayed_sequence},
        {"ops_since_last_checkpoint", s.ops_since_last_checkpoint},
        {"recovering", s.recovering},
        {"wal", {
            {"total_entries",     s.wal.total_entries},
            {"total_bytes",       s.wal.total_bytes},
            {"next_sequence",     s.wal.next_sequence},
            {"current_log_size",  s.wal.current_log_size}
        }}
    };
}