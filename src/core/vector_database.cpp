// src/core/vector_database.cpp
#include "vector_database.hpp"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>

VectorDatabase::VectorDatabase(size_t dimensions, bool useApproximate, size_t numTables, size_t numHashFunctions)
    : dimensions(dimensions), useApproximate(useApproximate) {
    distanceMetric = std::make_shared<EuclideanDistance>();
    kdTree = std::make_unique<KDTree>(dimensions, distanceMetric);
    lshIndex = std::make_unique<LSHIndex>(dimensions, numTables, numHashFunctions, distanceMetric.get());
}

void VectorDatabase::setDistanceMetric(std::shared_ptr<DistanceMetric> metric) {
    distanceMetric = std::move(metric);
    kdTree = std::make_unique<KDTree>(dimensions, distanceMetric);
}

void VectorDatabase::insert(const Vector& vector, const std::string& key) {
    if (vector.size() != dimensions) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    
    for (size_t i = 0; i < vector.size(); ++i) {
        if (std::isnan(vector[i])) {
            std::cout << "Warning: Vector " << key << " contains NaN values. Skipping insertion." << std::endl;
            return;
        }
    }
    
    kdTree->insert(vector, key);
    lshIndex->insert(vector, key);
    vectorMap[key] = vector;
}

void VectorDatabase::insert(const Vector& vector, const std::string& key, const std::string& metadata) {
    insert(vector, key);
    metadataMap[key] = metadata;
}

void VectorDatabase::batchInsert(const std::vector<Vector>& vectors, const std::vector<std::string>& keys) {
    if (vectors.size() != keys.size()) {
        throw std::invalid_argument("Number of vectors and keys must match");
    }
    for (size_t i = 0; i < vectors.size(); ++i) {
        insert(vectors[i], keys[i]);
    }
}

std::vector<std::pair<std::string, float>> VectorDatabase::similaritySearch(const Vector& query, size_t k) const {
    if (query.size() != dimensions) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }

    if (vectorMap.empty()) {
        return {};
    }
    
    if (useApproximate) {
        return lshIndex->search(query, k);
    } else {
        return kdTree->nearestNeighbors(query, k);
    }
}

std::vector<VectorDatabase::SearchResult> VectorDatabase::similaritySearchWithMetadata(const Vector& query, size_t k) const {
    auto rawResults = similaritySearch(query, k);
    std::vector<SearchResult> results;
    results.reserve(rawResults.size());
    
    for (const auto& [key, distance] : rawResults) {
        auto metaIt = metadataMap.find(key);
        results.push_back({
            key,
            distance,
            metaIt != metadataMap.end() ? metaIt->second : ""
        });
    }
    return results;
}

std::vector<std::vector<std::pair<std::string, float>>> VectorDatabase::batchSimilaritySearch(
    const std::vector<Vector>& queries, size_t k) const {
    std::vector<std::vector<std::pair<std::string, float>>> results;
    results.reserve(queries.size());
    for (const auto& query : queries) {
        results.push_back(similaritySearch(query, k));
    }
    return results;
}

void VectorDatabase::toggleApproximateSearch(bool useApprox) {
    useApproximate = useApprox;
}

std::string VectorDatabase::getMetadata(const std::string& key) const {
    auto it = metadataMap.find(key);
    if (it == metadataMap.end()) {
        throw std::runtime_error("Key not found in database");
    }
    return it->second;
}

void VectorDatabase::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint64_t file_dimensions, num_vectors;
    file.read(reinterpret_cast<char*>(&file_dimensions), sizeof(file_dimensions));
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));

    if (file_dimensions != dimensions) {
        throw std::runtime_error("File vector dimension does not match database dimension");
    }

    std::vector<float> buffer(dimensions);
    for (uint64_t i = 0; i < num_vectors; ++i) {
        uint32_t key_length;
        file.read(reinterpret_cast<char*>(&key_length), sizeof(key_length));
        std::string key(key_length, '\0');
        file.read(&key[0], key_length);

        file.read(reinterpret_cast<char*>(buffer.data()), dimensions * sizeof(float));
        Vector vec(buffer);

        uint32_t metadata_length;
        file.read(reinterpret_cast<char*>(&metadata_length), sizeof(metadata_length));
        std::string metadata(metadata_length, '\0');
        file.read(&metadata[0], metadata_length);

        insert(vec, key, metadata);
    }
}

void VectorDatabase::saveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    uint64_t num_vectors = vectorMap.size();
    file.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));

    for (const auto& [key, vector] : vectorMap) {
        uint32_t key_length = key.length();
        file.write(reinterpret_cast<const char*>(&key_length), sizeof(key_length));
        file.write(key.c_str(), key_length);

        file.write(reinterpret_cast<const char*>(vector.data_ptr()), dimensions * sizeof(float));

        const std::string& metadata = metadataMap.count(key) ? metadataMap.at(key) : "";
        uint32_t metadata_length = metadata.length();
        file.write(reinterpret_cast<const char*>(&metadata_length), sizeof(metadata_length));
        file.write(metadata.c_str(), metadata_length);
    }
}