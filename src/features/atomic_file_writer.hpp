
#pragma once

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>

/**
 * AtomicFileWriter provides safe file writing with atomic commits.
 * 
 * Features:
 * - Writes to temporary file first
 * - Atomic rename operation for commit
 * - fsync() for durability guarantees
 * - Checksum validation
 * - Exception safety
 */
class AtomicFileWriter {
private:
    std::string temp_filename;
    std::string final_filename;
    std::ofstream file;
    bool committed = false;
    bool aborted = false;
    
    // Generate temporary filename
    std::string generateTempFilename(const std::string& final_path);
    
    // Calculate simple checksum
    uint32_t calculateChecksum(const void* data, size_t size);
    
public:
    /**
     * Constructor - creates temporary file for writing
     * @param filename Final destination filename
     * @throws std::runtime_error if temporary file cannot be created
     */
    explicit AtomicFileWriter(const std::string& filename);
    
    /**
     * Destructor - aborts if not committed
     */
    ~AtomicFileWriter();
    
    // Non-copyable
    AtomicFileWriter(const AtomicFileWriter&) = delete;
    AtomicFileWriter& operator=(const AtomicFileWriter&) = delete;
    
    // Movable
    AtomicFileWriter(AtomicFileWriter&& other) noexcept;
    AtomicFileWriter& operator=(AtomicFileWriter&& other) noexcept;
    
    /**
     * Write data to the temporary file
     * @param data Pointer to data
     * @param size Number of bytes to write
     * @throws std::runtime_error if write fails
     */
    void write(const void* data, size_t size);
    
    /**
     * Write string to the temporary file
     * @param str String to write
     */
    void write(const std::string& str);
    
    /**
     * Write POD type to the temporary file
     * @param value Value to write
     */
    template<typename T>
    void write(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
        write(&value, sizeof(T));
    }
    
    /**
     * Atomically commit the file
     * - Flushes all data to disk
     * - Calls fsync() for durability
     * - Atomically renames temp file to final filename
     * @throws std::runtime_error if commit fails
     */
    void commit();
    
    /**
     * Abort the operation and clean up temporary file
     */
    void abort();
    
    /**
     * Check if the writer is in a valid state for writing
     * @return true if ready for writing, false if committed/aborted
     */
    bool isReady() const { return !committed && !aborted; }
    
    /**
     * Check if the operation was committed
     * @return true if committed
     */
    bool isCommitted() const { return committed; }
    
    /**
     * Get the final filename
     * @return Final destination filename
     */
    const std::string& getFilename() const { return final_filename; }
};


