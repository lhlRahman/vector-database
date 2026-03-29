#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

// Page-based mmap'd vector storage.
//
// File layout:
//   [Header — 4096 bytes (1 page)]
//   [Slot 0] [Slot 1] ... [Slot N-1]
//
// Each slot:
//   [SlotHeader — 384 bytes]
//   [float vector[dims]]
//   [padding to slot_size]
//
// Slot size is rounded up to 64-byte boundary for SIMD alignment.
// Vectors are directly accessible via pointer into mmap'd memory — zero copy.

class MMapStorage {
public:
    static constexpr size_t PAGE_SIZE = 4096;
    static constexpr size_t MAX_KEY_LEN = 120;
    static constexpr size_t MAX_META_LEN = 248;
    static constexpr size_t SLOT_HEADER_BYTES = 384; // see SlotHeader below

    enum SlotFlags : uint8_t {
        SLOT_EMPTY     = 0,
        SLOT_ACTIVE    = 1,
        SLOT_TOMBSTONE = 2,
    };

    struct FileHeader {
        char     magic[4];        // "VDBS"
        uint32_t version;         // 1
        uint32_t dimensions;
        uint32_t slot_size;       // SLOT_HEADER_BYTES + dims*4, rounded to 64
        uint64_t slot_capacity;   // total allocated slots
        uint64_t active_count;    // number of active (non-deleted) slots
        uint64_t next_slot_hint;  // hint for next free slot search
        char     reserved[PAGE_SIZE - 40];
    };
    static_assert(sizeof(FileHeader) == PAGE_SIZE);

    struct SlotHeader {
        uint8_t  flags;
        uint8_t  reserved[3];
        uint32_t key_len;
        char     key[MAX_KEY_LEN];       // 120 bytes
        uint32_t meta_len;
        char     metadata[MAX_META_LEN]; // 248 bytes
        // Total: 1+3+4+120+4+248 = 380, pad to 384
        uint8_t  pad[4];
        // Followed by: float vector[dims]
    };
    static_assert(sizeof(SlotHeader) == SLOT_HEADER_BYTES);

    // ── Lifecycle ──────────────────────────────────────────

    // path: file path. dims: vector dimensions. initial_capacity: pre-allocate slots.
    MMapStorage(const std::string& path, size_t dims, size_t initial_capacity = 4096);
    ~MMapStorage() noexcept;

    MMapStorage(const MMapStorage&) = delete;
    MMapStorage& operator=(const MMapStorage&) = delete;

    // Open existing file or create new one.
    void open();
    void close();

    // ── CRUD ───────────────────────────────────────────────

    // Insert vector+key+metadata. Returns slot ID, or UINT64_MAX on failure.
    uint64_t insert(const std::string& key, const float* vec, const std::string& metadata);

    // Update vector and/or metadata at a given slot.
    bool update(uint64_t slot_id, const float* vec, const std::string& metadata);

    // Mark slot as tombstone.
    bool remove(uint64_t slot_id);

    // ── Access (zero-copy where possible) ──────────────────

    // Returns pointer directly into mmap'd memory.
    // WARNING: Pointer is invalidated by grow(). Callers must hold an RCU
    // ReadGuard and only use pointers within the guard's scope. Writers that
    // trigger grow() hold a WriteGuard, which waits for all readers to exit.
    const float* vector_ptr(uint64_t slot_id) const;

    // Copy key/metadata out of slot.
    std::string get_key(uint64_t slot_id) const;
    std::string get_metadata(uint64_t slot_id) const;

    bool is_active(uint64_t slot_id) const;

    // ── Iteration ──────────────────────────────────────────

    uint64_t capacity() const;
    uint64_t active_count() const;
    size_t   dimensions() const;

    // Build key→slot_id map by scanning all active slots.
    std::unordered_map<std::string, uint64_t> build_key_index() const;

    // ── Maintenance ────────────────────────────────────────

    void sync(); // msync to disk

    // madvise hint: MADV_RANDOM for search, MADV_SEQUENTIAL for bulk load/rebuild
    void advise_random();
    void advise_sequential();
    void advise_willneed(uint64_t slot_start, uint64_t slot_count);

private:
    std::string path_;
    size_t dims_;
    size_t initial_capacity_;
    int    fd_ = -1;
    void*  mapped_ = nullptr;
    size_t mapped_size_ = 0;
    size_t slot_size_ = 0;

    FileHeader*       header();
    const FileHeader* header() const;
    SlotHeader*       slot(uint64_t id);
    const SlotHeader* slot(uint64_t id) const;
    float*            slot_vector(uint64_t id);
    const float*      slot_vector(uint64_t id) const;

    uint64_t find_free_slot();
    void     grow(size_t new_capacity);
    void     remap(size_t new_size);

    static size_t compute_slot_size(size_t dims);
    static size_t compute_file_size(size_t slot_size, size_t capacity);
};
