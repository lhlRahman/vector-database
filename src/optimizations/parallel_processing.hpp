#pragma once

#include <functional>
#include <string>
#include <vector>

// Forward-declare the classes to avoid circular dependencies if possible,
// but including the full header is necessary here for std::vector<Vector>, etc.
#include "../core/vector.hpp"
#include "../core/vector_database.hpp"

namespace parallel_ops {
    // Returns the number of inserts that succeeded (insert() returns false on
    // duplicate key / NaN / full segment); callers can detect partial failure.
    size_t batchInsert(VectorDatabase& db, const std::vector<Vector>& vectors, const std::vector<std::string>& keys);
    
    std::vector<std::vector<std::pair<std::string, float>>> batchSimilaritySearch(
        VectorDatabase& db, const std::vector<Vector>& queries, size_t k);

    void parallel_for_each(std::vector<int>& indices, const std::function<void(int)>& func);
    
    std::vector<float> parallel_transform(const std::vector<Vector>& queries, const Vector& centroid);
}