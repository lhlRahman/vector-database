#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

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

enum class FileSyncMode {
    Fsync,
    FullFsync,
};

struct FileSyncStatistics {
    // Counts only successful/failed regular-file synchronization calls. Directory
    // synchronization has a separate portability contract below.
    uint64_t fsync_successes{0};
    uint64_t full_fsync_successes{0};
    uint64_t failures{0};
};
}  // namespace vdb::io

// Atomic, durable file write helpers. Two callers (segment.cpp, the
// checkpoint path in atomic_persistence.cpp) used to have their own copies;
// they're consolidated here.
namespace vdb::io {

namespace detail {

struct FileSyncCounters {
    std::atomic<uint64_t> fsync_successes{0};
    std::atomic<uint64_t> full_fsync_successes{0};
    std::atomic<uint64_t> failures{0};
};

inline FileSyncCounters& file_sync_counters() {
    static FileSyncCounters counters;
    return counters;
}

using FileSyncCallOverride = int (*)(int, FileSyncMode);

inline std::atomic<FileSyncCallOverride>& file_sync_call_override() {
    static std::atomic<FileSyncCallOverride> override_call{nullptr};
    return override_call;
}

template <typename SyncCall>
inline void run_sync_call(int fd,
                          const std::filesystem::path& path,
                          const char* operation,
                          SyncCall&& sync_call) {
    for (;;) {
        if (sync_call(fd) == 0) return;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::runtime_error(std::string("fsync_file: ") + operation +
                                 " failed on " + path.string() + ": " +
                                 std::strerror(error));
    }
}

// Kept as a small dependency-injected unit so the strong-sync selection can be
// tested on every platform. A failed FullFsync branch never invokes plain_sync.
template <typename PlainSync, typename FullSync>
inline FileSyncMode sync_descriptor_with_calls(
    int fd,
    const std::filesystem::path& path,
    bool request_full_fsync,
    PlainSync&& plain_sync,
    FullSync&& full_sync) {
    if (request_full_fsync) {
        run_sync_call(fd, path, "F_FULLFSYNC", std::forward<FullSync>(full_sync));
        return FileSyncMode::FullFsync;
    }
    run_sync_call(fd, path, "fsync", std::forward<PlainSync>(plain_sync));
    return FileSyncMode::Fsync;
}

inline bool request_full_fsync_on_this_platform() {
#if defined(__APPLE__)
    return full_fsync_enabled().load(std::memory_order_relaxed);
#else
    return false;
#endif
}

inline int call_plain_fsync(int fd) {
    if (const auto override_call =
            file_sync_call_override().load(std::memory_order_relaxed)) {
        return override_call(fd, FileSyncMode::Fsync);
    }
    return ::fsync(fd);
}

inline int call_full_fsync(int fd) {
    if (const auto override_call =
            file_sync_call_override().load(std::memory_order_relaxed)) {
        return override_call(fd, FileSyncMode::FullFsync);
    }
#if defined(__APPLE__)
    return ::fcntl(fd, F_FULLFSYNC);
#else
    (void)fd;
    errno = ENOTSUP;
    return -1;
#endif
}

inline void record_file_sync_success(FileSyncMode mode) {
    auto& counters = file_sync_counters();
    if (mode == FileSyncMode::FullFsync) {
        counters.full_fsync_successes.fetch_add(1, std::memory_order_relaxed);
    } else {
        counters.fsync_successes.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void record_file_sync_failure() {
    file_sync_counters().failures.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace detail

// Process-wide telemetry reports which primitive actually completed. In
// particular, full_fsync_successes advances only after F_FULLFSYNC returns zero.
inline FileSyncStatistics file_sync_statistics() {
    auto& counters = detail::file_sync_counters();
    return FileSyncStatistics{
        counters.fsync_successes.load(std::memory_order_relaxed),
        counters.full_fsync_successes.load(std::memory_order_relaxed),
        counters.failures.load(std::memory_order_relaxed),
    };
}

namespace testing {

inline void set_file_sync_call_override(detail::FileSyncCallOverride override_call) {
    detail::file_sync_call_override().store(override_call, std::memory_order_relaxed);
}

}  // namespace testing

// Durability contract: these helpers now FAIL LOUDLY. Previously they swallowed
// open() failures and ignored fsync()'s return value, so an EIO/ENOSPC on sync
// was reported to callers as success — silently defeating the WAL / atomic-write
// durability guarantee. They now throw std::runtime_error if the data cannot be
// made durable.

inline FileSyncMode sync_file_descriptor(int fd, const std::filesystem::path& path) {
    try {
        const FileSyncMode mode = detail::sync_descriptor_with_calls(
            fd,
            path,
            detail::request_full_fsync_on_this_platform(),
            detail::call_plain_fsync,
            detail::call_full_fsync);
        detail::record_file_sync_success(mode);
        return mode;
    } catch (...) {
        detail::record_file_sync_failure();
        throw;
    }
}

inline FileSyncMode fsync_file(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        detail::record_file_sync_failure();
        throw std::runtime_error("fsync_file: cannot open " + path.string() +
                                 ": " + std::strerror(errno));
    }
    try {
        const FileSyncMode mode = sync_file_descriptor(fd, path);
        ::close(fd);
        return mode;
    } catch (...) {
        ::close(fd);
        throw;
    }
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
