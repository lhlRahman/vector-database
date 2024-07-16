// src/features/query_cache.cpp

#include "query_cache.hpp"

QueryCache::QueryCache(size_t capacity) : capacity(capacity) {}

bool QueryCache::get(const Vector& query, std::vector<std::pair<std::string, float>>& results) {
    auto it = cache.find(query);
    if (it != cache.end()) {
        lru_list.erase(it->second.first);
        lru_list.push_front(query);
        it->second.first = lru_list.begin();
        results = it->second.second.results;
        return true;
    }
    return false;
}

void QueryCache::put(const Vector& query, const std::vector<std::pair<std::string, float>>& results) {
    if (cache.size() == capacity) {
        auto last = lru_list.back();
        lru_list.pop_back();
        cache.erase(last);
    }
    lru_list.push_front(query);
    cache[query] = {lru_list.begin(), {query, results}};
}
