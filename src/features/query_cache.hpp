#pragma once

#include <list>
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
        Vector query;
        std::vector<std::pair<std::string, float>> results;
    };

    size_t capacity;
    std::list<Vector> lru_list;
    std::unordered_map<Vector, std::pair<std::list<Vector>::iterator, CacheEntry>> cache;
    
    // Statistics
    uint64_t hits{0};
    uint64_t misses{0};

public:
    explicit QueryCache(size_t capacity); // Marked explicit
    bool get(const Vector& query, std::vector<std::pair<std::string, float>>& results);
    void put(const Vector& query, const std::vector<std::pair<std::string, float>>& results);
    void clear();
    Statistics getStatistics() const;
};
