// include/vector_database.hpp

#pragma once
#include "../src/core/vector.hpp"
#include "../src/core/kd_tree.hpp"
#include "../src/algorithms/lsh_index.hpp"
#include "../src/utils/distance_metrics.hpp"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

class VectorDatabase {
private:
    std::unique_ptr<KDTree> kdTree;
    std::unique_ptr<LSHIndex> lshIndex;
    std::shared_ptr<DistanceMetric> distanceMetric;
    size_t dimensions;
    bool useApproximate;
    std::unordered_map<std::string, Vector> vectorMap;
    std::unordered_map<std::string, std::string> metadataMap;

public:
    struct SearchResult {
        std::string key;
        float distance;
        std::string metadata;
    };

    VectorDatabase(size_t dimensions, bool useApproximate = false, size_t numTables = 10, size_t numHashFunctions = 8);
    void setDistanceMetric(std::shared_ptr<DistanceMetric> metric);
    void insert(const Vector& vector, const std::string& key);
    void insert(const Vector& vector, const std::string& key, const std::string& metadata);
    void batchInsert(const std::vector<Vector>& vectors, const std::vector<std::string>& keys);
    std::vector<std::pair<std::string, float>> similaritySearch(const Vector& query, size_t k) const;
    std::vector<SearchResult> similaritySearchWithMetadata(const Vector& query, size_t k) const;
    std::vector<std::vector<std::pair<std::string, float>>> batchSimilaritySearch(const std::vector<Vector>& queries, size_t k) const;
    void toggleApproximateSearch(bool useApproximate);
    void loadFromFile(const std::string& filename);
    void saveToFile(const std::string& filename) const;
    std::string getMetadata(const std::string& key) const;
    size_t getDimensions() const { return dimensions; }
    bool isUsingApproximateSearch() const { return useApproximate; }
    const std::unordered_map<std::string, Vector>& getAllVectors() const { return vectorMap; }
};