#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

// Epoch-based Read-Copy-Update for lock-free reads.
//
// Readers: enter/exit critical section by writing their epoch to a slot.
//          No atomic RMW (read-modify-write), so no cache line bouncing.
//          Completely wait-free.
//
// Writers: advance the global epoch, then spin-wait until all readers
//          from the old epoch have exited. Serialized by a mutex.
//
// This is much faster than shared_mutex for read-heavy workloads because
// readers never contend with each other or with writers on the same cache line.
class EpochRCU {
    static constexpr size_t kMaxReaders = 128;
    static constexpr uint64_t kInactive = UINT64_MAX;

    struct alignas(64) ReaderSlot {
        std::atomic<uint64_t> epoch{kInactive};
    };

    std::array<ReaderSlot, kMaxReaders> slots_{};
    alignas(64) std::atomic<uint64_t> global_epoch_{0};
    alignas(64) std::atomic<size_t> next_slot_{0};
    std::mutex writer_mutex_;

    size_t acquire_slot() {
        // Hash thread ID to a slot. Collisions are safe — worst case,
        // a writer waits slightly longer than necessary.
        auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        return tid % kMaxReaders;
    }

public:
    // RAII read guard — enter/exit critical section
    class ReadGuard {
    public:
        explicit ReadGuard(EpochRCU& rcu) : rcu_(rcu) {
            slot_ = rcu_.acquire_slot();
            rcu_.slots_[slot_].epoch.store(
                rcu_.global_epoch_.load(std::memory_order_acquire),
                std::memory_order_release);
        }

        ~ReadGuard() {
            rcu_.slots_[slot_].epoch.store(kInactive, std::memory_order_release);
        }

        ReadGuard(const ReadGuard&) = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;

    private:
        EpochRCU& rcu_;
        size_t slot_;
    };

    // RAII write guard — exclusive access, waits for old-epoch readers
    class WriteGuard {
    public:
        explicit WriteGuard(EpochRCU& rcu) : rcu_(rcu) {
            rcu_.writer_mutex_.lock();
            // Advance epoch
            uint64_t old_epoch = rcu_.global_epoch_.fetch_add(1, std::memory_order_acq_rel);
            // Wait for all readers in the old epoch to exit
            for (size_t i = 0; i < kMaxReaders; ++i) {
                while (true) {
                    uint64_t reader_epoch = rcu_.slots_[i].epoch.load(std::memory_order_acquire);
                    if (reader_epoch == kInactive || reader_epoch > old_epoch) break;
                    std::this_thread::yield();
                }
            }
        }

        ~WriteGuard() {
            rcu_.writer_mutex_.unlock();
        }

        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;

    private:
        EpochRCU& rcu_;
    };
};
