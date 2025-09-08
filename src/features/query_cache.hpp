#pragma once

#include <list>
#include <string>
#include <unordered_map>
#include <utility> // Added for std::pair
#include <vector>

#include "../core/vector.hpp"

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
    explicit QueryCache(size_t capacity); // Marked explicit
    bool get(const Vector& query, std::vector<std::pair<std::string, float>>& results);
    void put(const Vector& query, const std::vector<std::pair<std::string, float>>& results);
};
