
#include <cstdio>    // Added for fclose
#include <cstring>   // Added for strerror
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept> // Added for std::runtime_error
#include <string>    // Added for std::string
#include <unistd.h>
#include <utility>   // Added for std::move
#include <filesystem>
#include "atomic_file_writer.hpp"

AtomicFileWriter::AtomicFileWriter(const std::string& filename) 
    : final_filename(filename) {
    
    // Create directory if it doesn't exist
    std::filesystem::path path(filename);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    
    temp_filename = generateTempFilename(filename);
    
    // Open temporary file for writing
    file.open(temp_filename, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create temporary file: " + temp_filename);
    }
}

AtomicFileWriter::~AtomicFileWriter() {
    if (!committed && !aborted) {
        // Auto-abort if not explicitly committed
        abort();
    }
}

AtomicFileWriter::AtomicFileWriter(AtomicFileWriter&& other) noexcept
    : temp_filename(std::move(other.temp_filename))
    , final_filename(std::move(other.final_filename))
    , file(std::move(other.file))
    , committed(other.committed)
    , aborted(other.aborted) {
    
    other.committed = true; // Prevent other from cleaning up
}

AtomicFileWriter& AtomicFileWriter::operator=(AtomicFileWriter&& other) noexcept {
    if (this != &other) {
        // Clean up current state
        if (!committed && !aborted) {
            abort();
        }
        
        temp_filename = std::move(other.temp_filename);
        final_filename = std::move(other.final_filename);
        file = std::move(other.file);
        committed = other.committed;
        aborted = other.aborted;
        
        other.committed = true; // Prevent other from cleaning up
    }
    return *this;
}

std::string AtomicFileWriter::generateTempFilename(const std::string& final_path) {
    std::filesystem::path path(final_path);
    std::string parent = path.parent_path().string();
    std::string stem = path.stem().string();
    std::string extension = path.extension().string();
    
    // Generate random suffix
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(100000, 999999);
    
    std::ostringstream oss;
    oss << parent << "/" << stem << ".tmp." << dis(gen) << extension;
    
    return oss.str();
}

uint32_t AtomicFileWriter::calculateChecksum(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t checksum = 0;
    
    for (size_t i = 0; i < size; ++i) {
        checksum = (checksum << 1) ^ bytes[i];
    }
    
    return checksum;
}

void AtomicFileWriter::write(const void* data, size_t size) {
    if (!isReady()) {
        throw std::runtime_error("Cannot write to committed or aborted file");
    }
    
    file.write(static_cast<const char*>(data), size);
    if (!file.good()) {
        throw std::runtime_error("Write operation failed");
    }
}

void AtomicFileWriter::write(const std::string& str) {
    write(str.c_str(), str.length());
}

void AtomicFileWriter::commit() {
    if (!isReady()) {
        throw std::runtime_error("Cannot commit already committed or aborted file");
    }
    
    // Flush all data to the file system
    file.flush();
    if (!file.good()) {
        throw std::runtime_error("Failed to flush data to disk");
    }
    
    // Get file descriptor for fsync (portable approach)
    // Close the file first to ensure all data is written
    file.close();
    
    // Open the file again with C-style file operations for fsync
    FILE* c_file = fopen(temp_filename.c_str(), "r+b");
    if (!c_file) {
        throw std::runtime_error("Cannot open file for fsync");
    }
    
    int fd = fileno(c_file);
    if (fd == -1) {
        fclose(c_file);
        throw std::runtime_error("Cannot get file descriptor for fsync");
    }
    
    // Force data to disk for durability
    if (fsync(fd) != 0) {
        fclose(c_file);
        throw std::runtime_error("fsync failed: " + std::string(strerror(errno)));
    }
    
    // Close the C file
    fclose(c_file);
    
    // Atomically rename temporary file to final filename
    if (std::rename(temp_filename.c_str(), final_filename.c_str()) != 0) {
        throw std::runtime_error("Atomic rename failed: " + std::string(strerror(errno)));
    }
    
    committed = true;
}

void AtomicFileWriter::abort() {
    if (committed) {
        return; // Nothing to abort
    }
    
    // Close file if open
    if (file.is_open()) {
        file.close();
    }
    
    // Remove temporary file
    if (!temp_filename.empty() && std::filesystem::exists(temp_filename)) {
        std::filesystem::remove(temp_filename);
    }
    
    aborted = true;
}
