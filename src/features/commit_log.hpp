#pragma once

#include <cstdint>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include "../core/vector.hpp"

enum class LogEntryType : uint32_t {
    INSERT     = 1,
    UPDATE     = 2,
    DELETE     = 3,
    CHECKPOINT = 4,
    COMMIT     = 5
};

struct LogEntry {
    uint64_t timestamp{0};         // wall clock (us)
    LogEntryType type{LogEntryType::COMMIT};
    uint64_t sequence_number{0};
    uint32_t checksum{0};
    uint32_t data_length{0};
    std::vector<uint8_t> data;

    LogEntry() = default;
    LogEntry(LogEntryType t, uint64_t seq, const std::vector<uint8_t>& d);

    std::vector<uint8_t> serialize() const;
    static LogEntry deserialize(const std::vector<uint8_t>& buffer);

    bool isValid() const;
private:
    uint32_t calculateChecksum() const;
};

// -------- Operations (inline strings; no string pool) ----------
struct InsertOperation {
    std::string key;
    Vector      vector;
    std::string metadata;

    std::vector<uint8_t> serialize() const;
    static InsertOperation deserialize(const std::vector<uint8_t>& data);
};

struct UpdateOperation {
    std::string key;
    Vector      vector;
    std::string metadata;

    std::vector<uint8_t> serialize() const;
    static UpdateOperation deserialize(const std::vector<uint8_t>& data);
};

struct DeleteOperation {
    std::string key;
    std::vector<uint8_t> serialize() const;
    static DeleteOperation deserialize(const std::vector<uint8_t>& data);
};

struct CheckpointOperation {
    uint64_t    checkpoint_sequence{0};
    std::string checkpoint_file;

    std::vector<uint8_t> serialize() const;
    static CheckpointOperation deserialize(const std::vector<uint8_t>& data);
};

// ---------------------------- CommitLog -----------------------------------
class CommitLog {
public:
    struct Statistics {
        uint64_t total_entries{0};
        uint64_t total_bytes{0};
        uint64_t next_sequence{1};
        uint64_t current_log_size{0};
    };

    CommitLog(const std::string& log_directory, size_t max_size, size_t max_files);
    ~CommitLog();

    void logInsert(const std::string& key, const Vector& vector, const std::string& metadata);
    void logUpdate(const std::string& key, const Vector& vector, const std::string& metadata);
    void logDelete(const std::string& key);
    void logCheckpoint(uint64_t checkpoint_sequence, const std::string& checkpoint_file);
    void logCommit();

    void flush();
    Statistics getStatistics() const;

    std::vector<LogEntry> readEntriesSince(uint64_t since_sequence) const;
    std::vector<LogEntry> readAllEntries() const;
    LogEntry findLatestCheckpoint() const;

    // hard reset (delete all WAL files and reopen as 000001)
    void reset();
    
    // Rotate to new log file (preserves sequence numbering)
    void rotateLog();
    
    // Clean up old log files
    void cleanupOldLogs();

private:
    std::string log_dir;
    std::string log_filename;
    size_t      max_log_size;
    size_t      max_log_files;

    mutable std::ofstream log_file;
    uint64_t   next_sequence_number;
    uint64_t   current_log_size;
    uint64_t   total_entries_written;
    uint64_t   total_bytes_written;

    std::string generateLogFilename(uint64_t sequence) const;
    void writeEntry(const LogEntry& entry);
};