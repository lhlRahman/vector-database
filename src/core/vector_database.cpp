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
    
    // Check for NaN values
    for (size_t i = 0; i < vector.size(); ++i) {
        if (std::isnan(vector[i])) {
            std::cout << "Warning: Vector " << key << " contains NaN values. Skipping insertion." << std::endl;
            return;
        }
    }
    
    std::cout << "Inserting " << key << ": ";
    for (size_t i = 0; i < std::min(vector.size(), size_t(5)); ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    kdTree->insert(vector, key);
    lshIndex->insert(vector, key);
    vectorMap[key] = vector;
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
        std::cout << "Warning: Database is empty. No similarity search performed." << std::endl;
        return {};
    }

    
    std::vector<std::pair<std::string, float>> results;
    
    if (useApproximate) {
        results = lshIndex->search(query, k);
    } else {
        results = kdTree->nearestNeighbors(query, k);
    }

    return results;
}

std::vector<std::vector<std::pair<std::string, float>>> VectorDatabase::batchSimilaritySearch(const std::vector<Vector>& queries, size_t k) const {
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

void VectorDatabase::loadFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read the number of vectors and their dimension
    uint64_t num_vectors, vector_dim;
file.read(reinterpret_cast<char*>(&num_vectors), sizeof(num_vectors));
    file.read(reinterpret_cast<char*>(&vector_dim), sizeof(vector_dim));

    if (vector_dim != dimensions) {
        throw std::runtime_error("File vector dimension does not match database dimension");
    }

    // Read vectors
    std::vector<float> buffer(vector_dim);
    std::string key;
    for (uint64_t i = 0; i < num_vectors; ++i) {
        // Read key length and key
        uint32_t key_length;
        file.read(reinterpret_cast<char*>(&key_length), sizeof(key_length));
        key.resize(key_length);
        file.read(&key[0], key_length);

        // Read vector data
        file.read(reinterpret_cast<char*>(buffer.data()), vector_dim * sizeof(float));

        // Create and insert the vector
        Vector vec(buffer);
        insert(vec, key);
    }

    if (!file) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    std::cout << "Loaded " << num_vectors << " vectors of dimension " << vector_dim << std::endl;
}

void VectorDatabase::saveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write the number of vectors and their dimension
    uint64_t num_vectors = vectorMap.size();
    uint64_t vector_dim = dimensions;
    file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
    file.write(reinterpret_cast<const char*>(&vector_dim), sizeof(vector_dim));

    // Write vectors
    for (const auto& [key, vector] : vectorMap) {
        // Write key length and key
        uint32_t key_length = key.length();
        file.write(reinterpret_cast<const char*>(&key_length), sizeof(key_length));
        file.write(key.c_str(), key_length);

        // Write vector data
        file.write(reinterpret_cast<const char*>(vector.data_ptr()), vector_dim * sizeof(float));
    }

    if (!file) {
        throw std::runtime_error("Error writing to file: " + filename);
    }

    std::cout << "Saved " << num_vectors << " vectors of dimension " << vector_dim << std::endl;
}