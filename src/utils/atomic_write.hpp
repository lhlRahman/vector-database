#pragma once

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <unistd.h>

// Atomic, durable file write helpers. Two callers (segment.cpp, the
// checkpoint path in atomic_persistence.cpp) used to have their own copies;
// they're consolidated here.
namespace vdb::io {

inline void fsync_file(const std::filesystem::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    ::fsync(fd);
    ::close(fd);
}

// fsync a directory so that prior rename/create/unlink operations within it
// are durable. POSIX requires this — without it, a rename can be lost on
// power failure even after the file itself was fsynced.
inline void fsync_dir(const std::filesystem::path& dir) {
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return;
    ::fsync(fd);
    ::close(fd);
}

// Write `path` atomically: stream to "path.tmp", flush+fsync, rename onto
// `path`, then fsync the parent directory. After this returns, the file
// content is durable under power loss; either the new content is visible
// or the old content is (no torn intermediate state).
inline void atomic_write(const std::filesystem::path& path,
                         const std::function<void(std::ostream&)>& writer) {
    std::filesystem::create_directories(path.parent_path());
    auto temp_path = path;
    temp_path += ".tmp";

    {
        std::ofstream os(temp_path, std::ios::binary | std::ios::trunc);
        if (!os.is_open()) {
            throw std::runtime_error("cannot open temp file: " + temp_path.string());
        }
        writer(os);
        os.flush();
        if (!os.good()) {
            throw std::runtime_error("failed to flush temp file: " + temp_path.string());
        }
    }

    fsync_file(temp_path);
    std::filesystem::rename(temp_path, path);
    fsync_dir(path.parent_path());
}

} // namespace vdb::io
