#pragma once
#include "vector.hpp"
#include "kd_tree.hpp"
#include "lsh_index.hpp"
#include "distance_metrics.hpp"
#include <memory>
#include <vector>
#include <string>

class VectorDatabase {
private:
    std::unique_ptr<KDTree> kdTree;
    std::unique_ptr<LSHIndex> lshIndex;
    std::shared_ptr<DistanceMetric> distanceMetric;
    size_t dimensions;
    bool useApproximate;

public:
    VectorDatabase(size_t dimensions, bool useApproximate = false, size_t numTables = 10, size_t numHashFunctions = 8);
    void setDistanceMetric(std::shared_ptr<DistanceMetric> metric);
    void insert(const Vector& vector, const std::string& key);
    void batchInsert(const std::vector<Vector>& vectors, const std::vector<std::string>& keys);
    std::vector<std::pair<std::string, float>> nearestNeighbors(const Vector& query, size_t k) const;
    std::vector<std::vector<std::pair<std::string, float>>> batchNearestNeighbors(const std::vector<Vector>& queries, size_t k) const;
    void toggleApproximateSearch(bool useApproximate);
};