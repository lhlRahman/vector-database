#include "query_cache.hpp"

QueryCache::QueryCache(size_t capacity) : capacity_(capacity) {}

bool QueryCache::get(const Vector& query, std::vector<std::pair<std::string, float>>& results) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = cache_.find(query);
    if (it != cache_.end()) {
        // Stale entry — generation mismatch
        if (it->second.second.generation != generation_.load(std::memory_order_relaxed)) {
            lru_list_.erase(it->second.first);
            cache_.erase(it);
            misses_++;
            return false;
        }
        // Cache hit - move to front (most recently used)
        lru_list_.erase(it->second.first);
        lru_list_.push_front(query);
        it->second.first = lru_list_.begin();
        results = it->second.second.results;
        hits_++;
        return true;
    }
    // Cache miss
    misses_++;
    return false;
}

void QueryCache::put(const Vector& query, const std::vector<std::pair<std::string, float>>& results) {
    std::lock_guard<std::mutex> lock(mtx_);
    uint64_t gen = generation_.load(std::memory_order_relaxed);

    // Check if already exists (update case)
    auto it = cache_.find(query);
    if (it != cache_.end()) {
        lru_list_.erase(it->second.first);
        lru_list_.push_front(query);
        it->second.first = lru_list_.begin();
        it->second.second.results = results;
        it->second.second.generation = gen;
        return;
    }

    // Evict least recently used if at capacity
    if (cache_.size() >= capacity_) {
        auto last = lru_list_.back();
        lru_list_.pop_back();
        cache_.erase(last);
    }

    // Insert new entry — no duplicate Vector in CacheEntry
    lru_list_.push_front(query);
    cache_[query] = {lru_list_.begin(), {results, gen}};
}

void QueryCache::invalidate() {
    // Lock-free: atomic increment invalidates all cached entries
    generation_.fetch_add(1, std::memory_order_release);
}

void QueryCache::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    cache_.clear();
    lru_list_.clear();
    hits_ = 0;
    misses_ = 0;
}

QueryCache::Statistics QueryCache::getStatistics() const {
    std::lock_guard<std::mutex> lock(mtx_);
    Statistics stats;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.current_size = cache_.size();
    stats.capacity = capacity_;
    return stats;
}
