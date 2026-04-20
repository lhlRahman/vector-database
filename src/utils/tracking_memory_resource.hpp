#pragma once

#include <atomic>
#include <cstddef>
#include <memory_resource>

class TrackingMemoryResource : public std::pmr::memory_resource {
public:
    struct Statistics {
        size_t allocation_calls;
        size_t deallocation_calls;
        size_t bytes_allocated;
        size_t bytes_deallocated;
        size_t bytes_outstanding;
        size_t peak_bytes_outstanding;
    };

    explicit TrackingMemoryResource(std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
        : upstream_(upstream) {}

    Statistics getStatistics() const {
        Statistics stats;
        stats.allocation_calls = allocation_calls_.load(std::memory_order_relaxed);
        stats.deallocation_calls = deallocation_calls_.load(std::memory_order_relaxed);
        stats.bytes_allocated = bytes_allocated_.load(std::memory_order_relaxed);
        stats.bytes_deallocated = bytes_deallocated_.load(std::memory_order_relaxed);
        stats.bytes_outstanding = bytes_outstanding_.load(std::memory_order_relaxed);
        stats.peak_bytes_outstanding = peak_bytes_outstanding_.load(std::memory_order_relaxed);
        return stats;
    }

    void resetStatistics() {
        allocation_calls_.store(0, std::memory_order_relaxed);
        deallocation_calls_.store(0, std::memory_order_relaxed);
        bytes_allocated_.store(0, std::memory_order_relaxed);
        bytes_deallocated_.store(0, std::memory_order_relaxed);
        bytes_outstanding_.store(0, std::memory_order_relaxed);
        peak_bytes_outstanding_.store(0, std::memory_order_relaxed);
    }

private:
    void* do_allocate(size_t bytes, size_t alignment) override {
        void* ptr = upstream_->allocate(bytes, alignment);
        allocation_calls_.fetch_add(1, std::memory_order_relaxed);
        bytes_allocated_.fetch_add(bytes, std::memory_order_relaxed);
        size_t outstanding = bytes_outstanding_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        updatePeak(outstanding);
        return ptr;
    }

    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
        deallocation_calls_.fetch_add(1, std::memory_order_relaxed);
        bytes_deallocated_.fetch_add(bytes, std::memory_order_relaxed);
        bytes_outstanding_.fetch_sub(bytes, std::memory_order_relaxed);
        upstream_->deallocate(ptr, bytes, alignment);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

    void updatePeak(size_t outstanding) {
        size_t current = peak_bytes_outstanding_.load(std::memory_order_relaxed);
        while (outstanding > current &&
               !peak_bytes_outstanding_.compare_exchange_weak(
                   current, outstanding, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    std::pmr::memory_resource* upstream_;
    std::atomic<size_t> allocation_calls_{0};
    std::atomic<size_t> deallocation_calls_{0};
    std::atomic<size_t> bytes_allocated_{0};
    std::atomic<size_t> bytes_deallocated_{0};
    std::atomic<size_t> bytes_outstanding_{0};
    std::atomic<size_t> peak_bytes_outstanding_{0};
};
