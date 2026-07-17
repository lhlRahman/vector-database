#pragma once

#include <atomic>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../core/vector.hpp"

class QueryCache {
public:
    struct Statistics {
        uint64_t hits{0};
        uint64_t misses{0};
        size_t current_size{0};
        size_t capacity{0};

        double hit_rate() const {
            uint64_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };

private:
    struct CacheEntry {
        std::vector<std::pair<std::string, float>> results;
        uint64_t generation;
        size_t k;  // how many neighbors these cached results can satisfy
    };

    size_t capacity_;
    std::atomic<uint64_t> generation_{0};
    std::list<Vector> lru_list_;
    std::unordered_map<Vector, std::pair<std::list<Vector>::iterator, CacheEntry>> cache_;
    mutable std::mutex mtx_;

    // Statistics
    uint64_t hits_{0};
    uint64_t misses_{0};

public:
    explicit QueryCache(size_t capacity);
    // A cache entry is only a hit if it was computed with k' >= the requested k
    // (top-k is a prefix of top-k'); otherwise it cannot satisfy the request.
    [[nodiscard]] bool get(const Vector& query, size_t k, std::vector<std::pair<std::string, float>>& results);
    void put(const Vector& query, size_t k, const std::vector<std::pair<std::string, float>>& results);
    void invalidate();
    void clear();
    Statistics getStatistics() const;
};
