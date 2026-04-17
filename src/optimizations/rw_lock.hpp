#pragma once

#include <shared_mutex>

// Reader/writer lock for VectorDatabase. Writers get exclusive access,
// readers share. Trivially a wrapper around std::shared_mutex; the only
// thing it adds is RAII guard types named to match the call-site usage.
//
// History: this used to be an epoch-based RCU (hence the original
// `EpochRCU` name). That implementation was unsound — WriteGuard only
// drained old-epoch readers, so readers entering after the epoch advance
// ran concurrently with the writer's in-place mutations of key_to_slot_,
// hnsw nodes, etc. ThreadSanitizer caught the resulting races on the
// existing "concurrent reads + writes" e2e test. Replacing it with a
// shared_mutex was the fix.
class RWLock {
    std::shared_mutex mutex_;

public:
    class ReadGuard {
    public:
        explicit ReadGuard(RWLock& lock) : lock_(lock.mutex_) {}
        ReadGuard(const ReadGuard&) = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;
    private:
        std::shared_lock<std::shared_mutex> lock_;
    };

    class WriteGuard {
    public:
        explicit WriteGuard(RWLock& lock) : lock_(lock.mutex_) {}
        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;
    private:
        std::unique_lock<std::shared_mutex> lock_;
    };
};
