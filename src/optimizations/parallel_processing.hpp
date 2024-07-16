// src/optimizations/parallel_processing.hpp

#pragma once
#include "../core/vector.hpp"
#include "../../include/vector_database.hpp"
#include <vector>
#include <string>
#include <functional>

namespace parallel_ops {
    void batchInsert(VectorDatabase& db, const std::vector<Vector>& vectors, const std::vector<std::string>& keys);
    std::vector<std::vector<std::pair<std::string, float>>> batchSimilaritySearch(
        const VectorDatabase& db, const std::vector<Vector>& queries, size_t k);

    void parallel_for_each(std::vector<int>& indices, std::function<void(int)>& func);
    std::vector<float> parallel_transform(const std::vector<Vector>& queries, const Vector& centroid);
}