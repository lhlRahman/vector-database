// src/features/query_cache.hpp

#pragma once
#include "../core/vector.hpp"
#include <unordered_map>
#include <list>
#include <vector>
#include <string>

class QueryCache {
private:
    struct CacheEntry {
        Vector query;
        std::vector<std::pair<std::string, float>> results;
    };

    size_t capacity;
    std::list<Vector> lru_list;
    std::unordered_map<Vector, std::pair<std::list<Vector>::iterator, CacheEntry>> cache;

public:
    QueryCache(size_t capacity);
    bool get(const Vector& query, std::vector<std::pair<std::string, float>>& results);
    void put(const Vector& query, const std::vector<std::pair<std::string, float>>& results);
};
