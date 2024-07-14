#include "vector_database.hpp"
#include <stdexcept>
#include <algorithm>

VectorDatabase::VectorDatabase(size_t dims, bool useApprox, size_t numTables, size_t numHashFunctions)
    : dimensions(dims), useApproximate(useApprox) {
    distanceMetric = std::make_shared<EuclideanDistance>();
    kdTree = std::make_unique<KDTree>(dims, distanceMetric);
    lshIndex = std::make_unique<LSHIndex>(dims, numTables, numHashFunctions);
}

void VectorDatabase::setDistanceMetric(std::shared_ptr<DistanceMetric> metric) {
    distanceMetric = std::move(metric);
    kdTree = std::make_unique<KDTree>(dimensions, distanceMetric);
}

void VectorDatabase::insert(const Vector& vector, const std::string& key) {
    if (vector.size() != dimensions) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    kdTree->insert(vector, key);
    lshIndex->insert(vector, key);
}

void VectorDatabase::batchInsert(const std::vector<Vector>& vectors, const std::vector<std::string>& keys) {
    if (vectors.size() != keys.size()) {
        throw std::invalid_argument("Number of vectors and keys must match");
    }
    
    for (size_t i = 0; i < vectors.size(); ++i) {
        insert(vectors[i], keys[i]);
    }
}

std::vector<std::pair<std::string, float>> VectorDatabase::nearestNeighbors(const Vector& query, size_t k) const {
    if (query.size() != dimensions) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }
    
    if (useApproximate) {
        return lshIndex->search(query, k, *distanceMetric);
    } else {
        std::vector<std::pair<std::string, float>> result;
        for (size_t i = 0; i < k; ++i) {
            std::string nearest = kdTree->nearest_neighbor(query);
            float distance = distanceMetric->distance(query, kdTree->getVector(nearest));
            result.emplace_back(nearest, distance);
            kdTree->removeTemporarily(nearest);
        }
        for (const auto& [key, _] : result) {
            kdTree->reinsert(key);
        }
        return result;
    }
}

std::vector<std::vector<std::pair<std::string, float>>> VectorDatabase::batchNearestNeighbors(const std::vector<Vector>& queries, size_t k) const {
    std::vector<std::vector<std::pair<std::string, float>>> results;
    results.reserve(queries.size());
    
    for (const auto& query : queries) {
        results.push_back(nearestNeighbors(query, k));
    }
    
    return results;
}

void VectorDatabase::toggleApproximateSearch(bool useApprox) {
    useApproximate = useApprox;
}
