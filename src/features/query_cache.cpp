#include <string>
#include <vector>

#include "query_cache.hpp"

QueryCache::QueryCache(size_t capacity) : capacity(capacity) {}

bool QueryCache::get(const Vector& query, std::vector<std::pair<std::string, float>>& results) {
    auto it = cache.find(query);
    if (it != cache.end()) {
        // Cache hit - move to front (most recently used)
        lru_list.erase(it->second.first);
        lru_list.push_front(query);
        it->second.first = lru_list.begin();
        results = it->second.second.results;
        hits++;
        return true;
    }
    // Cache miss
    misses++;
    return false;
}

void QueryCache::put(const Vector& query, const std::vector<std::pair<std::string, float>>& results) {
    // Check if already exists (update case)
    auto it = cache.find(query);
    if (it != cache.end()) {
        // Update existing entry
        lru_list.erase(it->second.first);
        lru_list.push_front(query);
        it->second.first = lru_list.begin();
        it->second.second.results = results;
        return;
    }
    
    // Evict least recently used if at capacity
    if (cache.size() >= capacity) {
        auto last = lru_list.back();
        lru_list.pop_back();
        cache.erase(last);
    }
    
    // Insert new entry
    lru_list.push_front(query);
    cache[query] = {lru_list.begin(), {query, results}};
}

void QueryCache::clear() {
    cache.clear();
    lru_list.clear();
    hits = 0;
    misses = 0;
}

QueryCache::Statistics QueryCache::getStatistics() const {
    Statistics stats;
    stats.hits = hits;
    stats.misses = misses;
    stats.current_size = cache.size();
    stats.capacity = capacity;
    return stats;
}
