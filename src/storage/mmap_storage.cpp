#include "mmap_storage.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

// Wrap a captured errno into a std::system_error. Callers can then catch by
// errno value (e.g. ENOENT) instead of grepping runtime_error::what().
namespace {
[[noreturn]] void throw_errno(int err, const std::string& what) {
    throw std::system_error(err, std::system_category(), what);
}
}

// ── Helpers ────────────────────────────────────────────────

size_t MMapStorage::compute_slot_size(size_t dims) {
    size_t raw = SLOT_HEADER_BYTES + dims * sizeof(float);
    return (raw + 63) & ~size_t(63); // round up to 64-byte boundary
}

size_t MMapStorage::compute_file_size(size_t slot_size, size_t capacity) {
    return PAGE_SIZE + slot_size * capacity;
}

// ── Accessors ──────────────────────────────────────────────

MMapStorage::FileHeader* MMapStorage::header() {
    return reinterpret_cast<FileHeader*>(mapped_);
}

const MMapStorage::FileHeader* MMapStorage::header() const {
    return reinterpret_cast<const FileHeader*>(mapped_);
}

MMapStorage::SlotHeader* MMapStorage::slot(uint64_t id) {
    auto* base = static_cast<uint8_t*>(mapped_) + PAGE_SIZE + id * slot_size_;
    return reinterpret_cast<SlotHeader*>(base);
}

const MMapStorage::SlotHeader* MMapStorage::slot(uint64_t id) const {
    auto* base = static_cast<const uint8_t*>(mapped_) + PAGE_SIZE + id * slot_size_;
    return reinterpret_cast<const SlotHeader*>(base);
}

float* MMapStorage::slot_vector(uint64_t id) {
    auto* base = static_cast<uint8_t*>(mapped_) + PAGE_SIZE + id * slot_size_;
    return reinterpret_cast<float*>(base + SLOT_HEADER_BYTES);
}

const float* MMapStorage::slot_vector(uint64_t id) const {
    auto* base = static_cast<const uint8_t*>(mapped_) + PAGE_SIZE + id * slot_size_;
    return reinterpret_cast<const float*>(base + SLOT_HEADER_BYTES);
}

// ── Lifecycle ──────────────────────────────────────────────

MMapStorage::MMapStorage(const std::string& path, size_t dims, size_t initial_capacity)
    : path_(path), dims_(dims), initial_capacity_(initial_capacity) {
    slot_size_ = compute_slot_size(dims);
}

MMapStorage::~MMapStorage() noexcept {
    try { close(); } catch (...) {}
}

void MMapStorage::open() {
    if (mapped_) return; // already open

    bool created = false;
    fd_ = ::open(path_.c_str(), O_RDWR | O_CLOEXEC, 0644);
    if (fd_ < 0) {
        // Create new file
        fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ < 0) {
            throw_errno(errno, "open(" + path_ + ")");
        }
        created = true;
    }


    if (created) {
        // Initialize new file
        size_t file_size = compute_file_size(slot_size_, initial_capacity_);
        if (ftruncate(fd_, static_cast<off_t>(file_size)) < 0) {
            int e = errno;
            ::close(fd_);
            fd_ = -1;  // reset so the destructor's close() doesn't double-close
            throw_errno(e, "ftruncate");
        }

        mapped_size_ = file_size;
        mapped_ = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped_ == MAP_FAILED) {
            int e = errno;
            mapped_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw_errno(e, "mmap");
        }

        // Zero the header, then fill it
        std::memset(mapped_, 0, PAGE_SIZE);
        auto* h = header();
        std::memcpy(h->magic, "VDBS", 4);
        h->version = 1;
        h->dimensions = static_cast<uint32_t>(dims_);
        h->slot_size = static_cast<uint32_t>(slot_size_);
        h->slot_capacity = initial_capacity_;
        h->active_count = 0;
        h->next_slot_hint = 0;

        // Zero all slots
        std::memset(static_cast<uint8_t*>(mapped_) + PAGE_SIZE, 0,
                    slot_size_ * initial_capacity_);
    } else {
        // Open existing file — read header to get sizes
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            int e = errno;
            ::close(fd_);
            fd_ = -1;
            throw_errno(e, "fstat");
        }
        mapped_size_ = static_cast<size_t>(st.st_size);

        mapped_ = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped_ == MAP_FAILED) {
            int e = errno;
            mapped_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw_errno(e, "mmap");
        }

        if (mapped_size_ < PAGE_SIZE) {
            close();
            throw std::runtime_error("storage file smaller than header");
        }

        auto* h = header();
        if (std::memcmp(h->magic, "VDBS", 4) != 0) {
            close();
            throw std::runtime_error("invalid file magic");
        }
        // Capture header values BEFORE close() unmaps the file. Otherwise
        // any reference to *h after close() dereferences a freed mapping.
        const uint32_t file_dims = h->dimensions;
        const uint32_t file_slot_size = h->slot_size;
        const uint64_t file_capacity = h->slot_capacity;
        if (file_dims != dims_) {
            close();
            throw std::runtime_error("dimension mismatch: file has " +
                                     std::to_string(file_dims) + ", expected " +
                                     std::to_string(dims_));
        }
        // Validate header geometry against the actual file size, so a truncated
        // or crafted file can't drive slot()/build_key_index() past the mapping.
        if (file_slot_size < SLOT_HEADER_BYTES + dims_ * sizeof(float) ||
            file_capacity > (mapped_size_ - PAGE_SIZE) / file_slot_size) {
            close();
            throw std::runtime_error("corrupt or truncated storage file");
        }
        slot_size_ = file_slot_size;
    }
}

void MMapStorage::close() {
    if (mapped_) {
        msync(mapped_, mapped_size_, MS_SYNC);
        munmap(mapped_, mapped_size_);
        mapped_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    mapped_size_ = 0;
}

// ── CRUD ───────────────────────────────────────────────────

uint64_t MMapStorage::find_free_slot() {
    auto* h = header();
    uint64_t cap = h->slot_capacity;
    uint64_t start = h->next_slot_hint;

    // Scan from hint forward, wrapping around
    for (uint64_t i = 0; i < cap; ++i) {
        uint64_t idx = (start + i) % cap;
        if (slot(idx)->flags != SLOT_ACTIVE) {
            h->next_slot_hint = idx + 1;
            return idx;
        }
    }
    return UINT64_MAX; // full
}

uint64_t MMapStorage::insert(const std::string& key, const float* vec, const std::string& metadata) {
    if (key.size() > MAX_KEY_LEN) {
        throw std::invalid_argument("key too long (max " + std::to_string(MAX_KEY_LEN) + ")");
    }
    if (metadata.size() > MAX_META_LEN) {
        throw std::invalid_argument("metadata too long (max " + std::to_string(MAX_META_LEN) + ")");
    }

    uint64_t id = find_free_slot();
    if (id == UINT64_MAX) {
        // Grow the file
        grow(header()->slot_capacity * 2);
        id = find_free_slot();
        if (id == UINT64_MAX) {
            throw std::runtime_error("failed to allocate slot after grow");
        }
    }

    auto* s = slot(id);
    // Write the payload FIRST, then publish the slot by setting flags = ACTIVE
    // last. On an unclean crash a slot is only ever seen as ACTIVE once its key/
    // metadata/vector are written (recovery ignores non-ACTIVE slots).
    s->key_len = static_cast<uint32_t>(key.size());
    std::memset(s->key, 0, MAX_KEY_LEN);
    std::memcpy(s->key, key.data(), key.size());
    s->meta_len = static_cast<uint32_t>(metadata.size());
    std::memset(s->metadata, 0, MAX_META_LEN);
    std::memcpy(s->metadata, metadata.data(), metadata.size());
    std::memcpy(slot_vector(id), vec, dims_ * sizeof(float));

    std::atomic_thread_fence(std::memory_order_release);
    s->flags = SLOT_ACTIVE;  // commit marker, written last

    header()->active_count++;
    return id;
}

bool MMapStorage::update(uint64_t slot_id, const float* vec, const std::string& metadata) {
    if (metadata.size() > MAX_META_LEN) {
        throw std::invalid_argument("metadata too long (max " + std::to_string(MAX_META_LEN) + ")");
    }
    if (slot_id >= header()->slot_capacity) return false;
    auto* s = slot(slot_id);
    if (s->flags != SLOT_ACTIVE) return false;

    std::memcpy(slot_vector(slot_id), vec, dims_ * sizeof(float));

    s->meta_len = static_cast<uint32_t>(metadata.size());
    std::memset(s->metadata, 0, MAX_META_LEN);
    std::memcpy(s->metadata, metadata.data(), metadata.size());
    return true;
}

bool MMapStorage::remove(uint64_t slot_id) {
    if (slot_id >= header()->slot_capacity) return false;
    auto* s = slot(slot_id);
    if (s->flags != SLOT_ACTIVE) return false;

    s->flags = SLOT_TOMBSTONE;
    header()->active_count--;

    // Update hint if this slot is earlier
    if (slot_id < header()->next_slot_hint) {
        header()->next_slot_hint = slot_id;
    }
    return true;
}

// ── Access ─────────────────────────────────────────────────

const float* MMapStorage::vector_ptr(uint64_t slot_id) const {
    return slot_vector(slot_id);
}

std::string MMapStorage::get_key(uint64_t slot_id) const {
    auto* s = slot(slot_id);
    // Clamp the stored length to the fixed field size: a corrupt/crafted file
    // could set key_len far beyond MAX_KEY_LEN and cause an OOB read.
    return std::string(s->key, std::min<size_t>(s->key_len, MAX_KEY_LEN));
}

std::string MMapStorage::get_metadata(uint64_t slot_id) const {
    auto* s = slot(slot_id);
    return std::string(s->metadata, std::min<size_t>(s->meta_len, MAX_META_LEN));
}

bool MMapStorage::is_active(uint64_t slot_id) const {
    if (slot_id >= header()->slot_capacity) return false;
    return slot(slot_id)->flags == SLOT_ACTIVE;
}

// ── Iteration ──────────────────────────────────────────────

uint64_t MMapStorage::capacity() const { return header()->slot_capacity; }
uint64_t MMapStorage::active_count() const { return header()->active_count; }
size_t   MMapStorage::dimensions() const { return dims_; }

std::unordered_map<std::string, uint64_t> MMapStorage::build_key_index() const {
    std::unordered_map<std::string, uint64_t> index;
    uint64_t cap = header()->slot_capacity;
    index.reserve(header()->active_count);
    for (uint64_t i = 0; i < cap; ++i) {
        if (slot(i)->flags == SLOT_ACTIVE) {
            index[get_key(i)] = i;
        }
    }
    return index;
}

// ── Maintenance ────────────────────────────────────────────

void MMapStorage::sync() {
    if (mapped_) {
        msync(mapped_, mapped_size_, MS_SYNC);
    }
}

void MMapStorage::advise_random() {
    if (mapped_) {
        madvise(mapped_, mapped_size_, MADV_RANDOM);
    }
}

void MMapStorage::advise_sequential() {
    if (mapped_) {
        madvise(mapped_, mapped_size_, MADV_SEQUENTIAL);
    }
}

void MMapStorage::advise_willneed(uint64_t slot_start, uint64_t slot_count) {
    if (!mapped_ || slot_count == 0) return;
    auto* base = static_cast<uint8_t*>(mapped_) + PAGE_SIZE + slot_start * slot_size_;
    size_t len = slot_count * slot_size_;
    // Clamp to mapped region. If base is already past the end (slot_start beyond
    // capacity), map_end - base would be a negative ptrdiff cast to a huge size_t.
    auto* map_end = static_cast<uint8_t*>(mapped_) + mapped_size_;
    if (base >= map_end) return;
    size_t max_len = static_cast<size_t>(map_end - base);
    if (len > max_len) len = max_len;
    madvise(base, len, MADV_WILLNEED);
}

// ── Growth ─────────────────────────────────────────────────

void MMapStorage::grow(size_t new_capacity) {
    auto* h = header();
    if (new_capacity <= h->slot_capacity) return;

    size_t old_capacity = h->slot_capacity;
    size_t new_size = compute_file_size(slot_size_, new_capacity);

    // Unmap current
    msync(mapped_, mapped_size_, MS_SYNC);
    munmap(mapped_, mapped_size_);
    mapped_ = nullptr;

    // Extend file
    if (ftruncate(fd_, static_cast<off_t>(new_size)) < 0) {
        throw_errno(errno, "ftruncate (grow)");
    }

    // Remap
    mapped_size_ = new_size;
    mapped_ = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        throw_errno(errno, "mmap (grow)");
    }

    // Zero new slots
    size_t old_data = PAGE_SIZE + slot_size_ * old_capacity;
    size_t new_data = slot_size_ * (new_capacity - old_capacity);
    std::memset(static_cast<uint8_t*>(mapped_) + old_data, 0, new_data);

    header()->slot_capacity = new_capacity;
}
