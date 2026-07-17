#pragma once

#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace vdb::io {
// On macOS plain fsync() only pushes to the drive, NOT through its write cache —
// true power-loss durability needs fcntl(F_FULLFSYNC), which is much slower.
// Off by default (dev/test speed + behavior parity); enable for honest durable
// writes / to measure true fsync cost. No-op on non-Apple platforms.
inline std::atomic<bool>& full_fsync_enabled() {
    static std::atomic<bool> flag{false};
    return flag;
}
inline void set_full_fsync(bool on) { full_fsync_enabled().store(on, std::memory_order_relaxed); }
}  // namespace vdb::io

// Atomic, durable file write helpers. Two callers (segment.cpp, the
// checkpoint path in atomic_persistence.cpp) used to have their own copies;
// they're consolidated here.
namespace vdb::io {

// Durability contract: these helpers now FAIL LOUDLY. Previously they swallowed
// open() failures and ignored fsync()'s return value, so an EIO/ENOSPC on sync
// was reported to callers as success — silently defeating the WAL / atomic-write
// durability guarantee. They now throw std::runtime_error if the data cannot be
// made durable.

inline void fsync_file(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("fsync_file: cannot open " + path.string() +
                                 ": " + std::strerror(errno));
    }
#if defined(__APPLE__)
    if (full_fsync_enabled().load(std::memory_order_relaxed)) {
        if (::fcntl(fd, F_FULLFSYNC) == 0) { ::close(fd); return; }
        // F_FULLFSYNC unsupported on this fs — fall through to plain fsync.
    }
#endif
    while (::fsync(fd) != 0) {
        if (errno == EINTR) continue;  // interrupted by signal — retry
        int e = errno;
        ::close(fd);
        throw std::runtime_error("fsync_file: fsync failed on " + path.string() +
                                 ": " + std::strerror(e));
    }
    ::close(fd);
}

// fsync a directory so that prior rename/create/unlink operations within it
// are durable. POSIX requires this — without it, a rename can be lost on
// power failure even after the file itself was fsynced.
inline void fsync_dir(const std::filesystem::path& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("fsync_dir: cannot open " + dir.string() +
                                 ": " + std::strerror(errno));
    }
    while (::fsync(fd) != 0) {
        if (errno == EINTR) continue;
        // Some filesystems legitimately do not support fsync on a directory
        // handle. Tolerate only those specific cases; every other error means
        // the directory metadata may not be durable, which we must surface.
        if (errno == EINVAL || errno == ENOTSUP) break;
        int e = errno;
        ::close(fd);
        throw std::runtime_error("fsync_dir: fsync failed on " + dir.string() +
                                 ": " + std::strerror(e));
    }
    ::close(fd);
}

// Build a temp path that is unique per process and per call, so two concurrent
// atomic_write() calls targeting the same final path cannot clobber each
// other's temp file or race on the rename. (The previous fixed "<path>.tmp"
// collided under concurrent flush()/checkpoint().)
inline std::filesystem::path make_temp_path(const std::filesystem::path& path) {
    static std::atomic<uint64_t> counter{0};
    uint64_t n = counter.fetch_add(1, std::memory_order_relaxed);
    auto tmp = path;
    tmp += ".tmp." + std::to_string(::getpid()) + "." + std::to_string(n);
    return tmp;
}

// Write `path` atomically: stream to a unique "path.tmp.<pid>.<n>", flush+fsync,
// rename onto `path`, then fsync the parent directory. After this returns, the
// file content is durable under power loss; either the new content is visible
// or the old content is (no torn intermediate state).
inline void atomic_write(const std::filesystem::path& path,
                         const std::function<void(std::ostream&)>& writer) {
    std::filesystem::create_directories(path.parent_path());
    auto temp_path = make_temp_path(path);

    {
        std::ofstream os(temp_path, std::ios::binary | std::ios::trunc);
        if (!os.is_open()) {
            throw std::runtime_error("cannot open temp file: " + temp_path.string());
        }
        writer(os);
        os.flush();
        if (!os.good()) {
            std::error_code rm_ec;
            std::filesystem::remove(temp_path, rm_ec);
            throw std::runtime_error("failed to flush temp file: " + temp_path.string());
        }
    }

    fsync_file(temp_path);
    std::filesystem::rename(temp_path, path);
    fsync_dir(path.parent_path());
}

} // namespace vdb::io
