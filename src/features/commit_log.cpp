// src/features/commit_log.cpp
#include "commit_log.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <chrono>

// Helper function for timestamp
namespace {
inline uint64_t now_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}
} // namespace


LogEntry::LogEntry(LogEntryType t, uint64_t seq, const std::vector<uint8_t>& d)
    : timestamp(now_us()), type(t), sequence_number(seq), data_length(static_cast<uint32_t>(d.size())), data(d) {
    checksum = calculateChecksum();
}

uint32_t LogEntry::calculateChecksum() const {
    uint32_t crc = 0;
    crc ^= static_cast<uint32_t>(timestamp);
    crc ^= static_cast<uint32_t>(type);
    crc ^= static_cast<uint32_t>(sequence_number);
    crc ^= data_length;
    for (uint8_t byte : data) {
        crc ^= byte;
    }
    return crc;
}

bool LogEntry::isValid() const {
    return checksum == calculateChecksum();
}

std::vector<uint8_t> LogEntry::serialize() const {
    std::vector<uint8_t> buffer;
    size_t total_size = sizeof(timestamp) + sizeof(type) + sizeof(sequence_number) + 
                       sizeof(checksum) + sizeof(data_length) + data.size();
    buffer.reserve(total_size);
    
    // Serialize fields
    auto append = [&buffer](const void* ptr, size_t size) {
        const uint8_t* bytes = static_cast<const uint8_t*>(ptr);
        buffer.insert(buffer.end(), bytes, bytes + size);
    };
    
    append(&timestamp, sizeof(timestamp));
    append(&type, sizeof(type));
    append(&sequence_number, sizeof(sequence_number));
    append(&checksum, sizeof(checksum));
    append(&data_length, sizeof(data_length));
    append(data.data(), data.size());
    
    return buffer;
}

LogEntry LogEntry::deserialize(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < sizeof(uint64_t) + sizeof(LogEntryType) + sizeof(uint64_t) + 
                       sizeof(uint32_t) + sizeof(uint32_t)) {
        return LogEntry(); // Invalid entry
    }
    
    LogEntry entry;
    size_t offset = 0;
    
    auto read = [&buffer, &offset](void* dest, size_t size) -> bool {
        if (offset + size > buffer.size()) return false;
        std::memcpy(dest, buffer.data() + offset, size);
        offset += size;
        return true;
    };
    
    if (!read(&entry.timestamp, sizeof(entry.timestamp)) ||
        !read(&entry.type, sizeof(entry.type)) ||
        !read(&entry.sequence_number, sizeof(entry.sequence_number)) ||
        !read(&entry.checksum, sizeof(entry.checksum)) ||
        !read(&entry.data_length, sizeof(entry.data_length))) {
        return LogEntry(); // Invalid entry
    }
    
    if (entry.data_length > 0) {
        if (offset + entry.data_length > buffer.size()) {
            return LogEntry(); // Invalid entry
        }
        entry.data.resize(entry.data_length);
        if (!read(entry.data.data(), entry.data_length)) {
            return LogEntry(); // Invalid entry
        }
    }
    
    return entry;
}

// ========================== Operation Serialization ==========================

std::vector<uint8_t> InsertOperation::serialize() const {
    std::vector<uint8_t> buffer;
    
    auto writeString = [&buffer](const std::string& s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(&len), 
                     reinterpret_cast<const uint8_t*>(&len) + sizeof(len));
        buffer.insert(buffer.end(), s.begin(), s.end());
    };
    
    auto writeVector = [&buffer](const Vector& v) {
        uint32_t dims = static_cast<uint32_t>(v.size());
        buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(&dims), 
                     reinterpret_cast<const uint8_t*>(&dims) + sizeof(dims));
        const float* data = v.data_ptr();
        buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(data), 
                     reinterpret_cast<const uint8_t*>(data) + dims * sizeof(float));
    };
    
    writeString(key);
    writeVector(vector);
    writeString(metadata);
    
    return buffer;
}

InsertOperation InsertOperation::deserialize(const std::vector<uint8_t>& data) {
    InsertOperation op;
    size_t offset = 0;
    
    auto readString = [&data, &offset](std::string& s) -> bool {
        if (offset + sizeof(uint32_t) > data.size()) return false;
        uint32_t len;
        std::memcpy(&len, data.data() + offset, sizeof(len));
        offset += sizeof(len);
        
        if (offset + len > data.size()) return false;
        s.assign(reinterpret_cast<const char*>(data.data() + offset), len);
        offset += len;
        return true;
    };
    
    auto readVector = [&data, &offset](Vector& v) -> bool {
        if (offset + sizeof(uint32_t) > data.size()) return false;
        uint32_t dims;
        std::memcpy(&dims, data.data() + offset, sizeof(dims));
        offset += sizeof(dims);
        
        if (offset + dims * sizeof(float) > data.size()) return false;
        std::vector<float> vec_data(dims);
        std::memcpy(vec_data.data(), data.data() + offset, dims * sizeof(float));
        offset += dims * sizeof(float);
        v = Vector(vec_data);
        return true;
    };
    
    if (!readString(op.key) || !readVector(op.vector) || !readString(op.metadata)) {
        return InsertOperation(); // Invalid
    }
    
    return op;
}

std::vector<uint8_t> UpdateOperation::serialize() const {
    return InsertOperation{key, vector, metadata}.serialize();
}

UpdateOperation UpdateOperation::deserialize(const std::vector<uint8_t>& data) {
    auto insert_op = InsertOperation::deserialize(data);
    return UpdateOperation{insert_op.key, insert_op.vector, insert_op.metadata};
}

std::vector<uint8_t> DeleteOperation::serialize() const {
    std::vector<uint8_t> buffer;
    uint32_t len = static_cast<uint32_t>(key.size());
    buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(&len), 
                 reinterpret_cast<const uint8_t*>(&len) + sizeof(len));
    buffer.insert(buffer.end(), key.begin(), key.end());
    return buffer;
}

DeleteOperation DeleteOperation::deserialize(const std::vector<uint8_t>& data) {
    DeleteOperation op;
    if (data.size() < sizeof(uint32_t)) return op;
    
    uint32_t len;
    std::memcpy(&len, data.data(), sizeof(len));
    
    if (sizeof(uint32_t) + len > data.size()) return op;
    
    op.key.assign(reinterpret_cast<const char*>(data.data() + sizeof(uint32_t)), len);
    return op;
}

std::vector<uint8_t> CheckpointOperation::serialize() const {
    std::vector<uint8_t> buffer;
    
    buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(&checkpoint_sequence),
                 reinterpret_cast<const uint8_t*>(&checkpoint_sequence) + sizeof(checkpoint_sequence));
    
    uint32_t len = static_cast<uint32_t>(checkpoint_file.size());
    buffer.insert(buffer.end(), reinterpret_cast<const uint8_t*>(&len),
                 reinterpret_cast<const uint8_t*>(&len) + sizeof(len));
    buffer.insert(buffer.end(), checkpoint_file.begin(), checkpoint_file.end());
    
    return buffer;
}

CheckpointOperation CheckpointOperation::deserialize(const std::vector<uint8_t>& data) {
    CheckpointOperation op;
    if (data.size() < sizeof(uint64_t) + sizeof(uint32_t)) return op;
    
    size_t offset = 0;
    std::memcpy(&op.checkpoint_sequence, data.data() + offset, sizeof(op.checkpoint_sequence));
    offset += sizeof(op.checkpoint_sequence);
    
    uint32_t len;
    std::memcpy(&len, data.data() + offset, sizeof(len));
    offset += sizeof(len);
    
    if (offset + len > data.size()) return op;
    
    op.checkpoint_file.assign(reinterpret_cast<const char*>(data.data() + offset), len);
    return op;
}

// ========================== CommitLog Implementation ==========================

CommitLog::CommitLog(const std::string& log_directory, size_t max_size, size_t max_files)
    : log_dir(log_directory), max_log_size(max_size), max_log_files(max_files),
      next_sequence_number(1), current_log_size(0), total_entries_written(0), total_bytes_written(0) {
    
    // Create log directory if it doesn't exist
    std::filesystem::create_directories(log_dir);
    
    // Generate initial log filename
    log_filename = generateLogFilename(next_sequence_number);
    
    // Open log file for writing
    log_file.open(log_filename, std::ios::binary | std::ios::app);
    if (!log_file.is_open()) {
        throw std::runtime_error("Cannot open log file: " + log_filename);
    }
    
    // Get current file size
    log_file.seekp(0, std::ios::end);
    current_log_size = log_file.tellp();
    log_file.seekp(0, std::ios::end); // Stay at end for appending
}

CommitLog::~CommitLog() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

std::string CommitLog::generateLogFilename(uint64_t sequence) const {
    std::ostringstream oss;
    oss << log_dir << "/commit.log." << std::setfill('0') << std::setw(6) << sequence;
    return oss.str();
}

void CommitLog::writeEntry(const LogEntry& entry) {
    auto serialized = entry.serialize();
    
    log_file.write(reinterpret_cast<const char*>(serialized.data()), serialized.size());
    log_file.flush();
    
    current_log_size += serialized.size();
    total_entries_written++;
    total_bytes_written += serialized.size();
    
    // Check if we need to rotate the log
    if (current_log_size >= max_log_size) {
        rotateLog();
    }
}

void CommitLog::rotateLog() {
    log_file.close();
    
    // Generate new log filename
    next_sequence_number++;
    log_filename = generateLogFilename(next_sequence_number);
    
    // Open new log file
    log_file.open(log_filename, std::ios::binary | std::ios::app);
    if (!log_file.is_open()) {
        throw std::runtime_error("Cannot open new log file: " + log_filename);
    }
    
    current_log_size = 0;
    
    // Clean up old logs
    cleanupOldLogs();
}

void CommitLog::cleanupOldLogs() {
    std::vector<std::string> log_files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(log_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("commit.log.") == 0) {
                    log_files.push_back(entry.path().string());
                }
            }
        }
        
        // Sort by filename (which includes sequence number)
        std::sort(log_files.begin(), log_files.end());
        
        // Remove oldest files if we exceed max_log_files
        while (log_files.size() > max_log_files) {
            std::filesystem::remove(log_files.front());
            log_files.erase(log_files.begin());
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Error cleaning up old log files: " << e.what() << std::endl;
    }
}

void CommitLog::logInsert(const std::string& key, const Vector& vector, const std::string& metadata) {
    InsertOperation op{key, vector, metadata};
    LogEntry entry(LogEntryType::INSERT, next_sequence_number++, op.serialize());
    writeEntry(entry);
}

void CommitLog::logUpdate(const std::string& key, const Vector& vector, const std::string& metadata) {
    UpdateOperation op{key, vector, metadata};
    LogEntry entry(LogEntryType::UPDATE, next_sequence_number++, op.serialize());
    writeEntry(entry);
}

void CommitLog::logDelete(const std::string& key) {
    DeleteOperation op{key};
    LogEntry entry(LogEntryType::DELETE, next_sequence_number++, op.serialize());
    writeEntry(entry);
}

void CommitLog::logCheckpoint(uint64_t checkpoint_sequence, const std::string& checkpoint_file) {
    CheckpointOperation op{checkpoint_sequence, checkpoint_file};
    LogEntry entry(LogEntryType::CHECKPOINT, next_sequence_number++, op.serialize());
    writeEntry(entry);
}

void CommitLog::logCommit() {
    LogEntry entry(LogEntryType::COMMIT, next_sequence_number++, std::vector<uint8_t>());
    writeEntry(entry);
}

void CommitLog::flush() {
    if (log_file.is_open()) {
        log_file.flush();
    }
}

CommitLog::Statistics CommitLog::getStatistics() const {
    Statistics stats;
    stats.total_entries = total_entries_written;
    stats.total_bytes = total_bytes_written;
    stats.next_sequence = next_sequence_number;
    stats.current_log_size = current_log_size;
    return stats;
}

std::vector<LogEntry> CommitLog::readEntriesSince(uint64_t since_sequence) const {
    std::vector<LogEntry> entries;
    
    try {
        // Find all log files
        std::vector<std::string> log_files;
        for (const auto& entry : std::filesystem::directory_iterator(log_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("commit.log.") == 0) {
                    log_files.push_back(entry.path().string());
                }
            }
        }
        
        std::sort(log_files.begin(), log_files.end());
        
        // Read entries from each log file
        for (const auto& file_path : log_files) {
            std::ifstream file(file_path, std::ios::binary);
            if (!file.is_open()) continue;
            
            // Read file content
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            
            std::vector<uint8_t> buffer(file_size);
            file.read(reinterpret_cast<char*>(buffer.data()), file_size);
            
            // Parse entries
            size_t offset = 0;
            while (offset < buffer.size()) {
                if (offset + sizeof(uint64_t) + sizeof(LogEntryType) + sizeof(uint64_t) + 
                    sizeof(uint32_t) + sizeof(uint32_t) > buffer.size()) {
                    break; // Not enough data for header
                }
                
                // Read entry header to get data length
                uint32_t data_length;
                std::memcpy(&data_length, buffer.data() + offset + sizeof(uint64_t) + 
                           sizeof(LogEntryType) + sizeof(uint64_t) + sizeof(uint32_t), sizeof(data_length));
                
                size_t entry_size = sizeof(uint64_t) + sizeof(LogEntryType) + sizeof(uint64_t) + 
                                   sizeof(uint32_t) + sizeof(uint32_t) + data_length;
                
                if (offset + entry_size > buffer.size()) {
                    break; // Not enough data for complete entry
                }
                
                std::vector<uint8_t> entry_data(buffer.begin() + offset, buffer.begin() + offset + entry_size);
                LogEntry entry = LogEntry::deserialize(entry_data);
                
                if (entry.isValid() && entry.sequence_number >= since_sequence) {
                    entries.push_back(entry);
                }
                
                offset += entry_size;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading log entries: " << e.what() << std::endl;
    }
    
    return entries;
}

std::vector<LogEntry> CommitLog::readAllEntries() const {
    return readEntriesSince(0);
}

LogEntry CommitLog::findLatestCheckpoint() const {
    auto entries = readAllEntries();
    
    // Find the most recent checkpoint entry
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        if (it->type == LogEntryType::CHECKPOINT) {
            return *it;
        }
    }
    
    return LogEntry(); // No checkpoint found
}

void CommitLog::reset() {
    // Close current log file
    if (log_file.is_open()) {
        log_file.close();
    }
    
    try {
        // Remove all existing log files
        for (const auto& entry : std::filesystem::directory_iterator(log_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find("commit.log.") == 0) {
                    std::filesystem::remove(entry.path());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Error removing old log files: " << e.what() << std::endl;
    }
    
    // Reset state
    next_sequence_number = 1;
    current_log_size = 0;
    total_entries_written = 0;
    total_bytes_written = 0;
    
    // Create new log file
    log_filename = generateLogFilename(next_sequence_number);
    log_file.open(log_filename, std::ios::binary | std::ios::app);
    
    if (!log_file.is_open()) {
        throw std::runtime_error("Cannot create new log file after reset: " + log_filename);
    }
}