// src/features/persistence.cpp
#include "persistence.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

void Persistence::save(const VectorDatabase& db, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Save database metadata
    size_t dimensions = db.getDimensions();
    bool useApproximate = db.isUsingApproximateSearch();
    file.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    file.write(reinterpret_cast<const char*>(&useApproximate), sizeof(useApproximate));

    // Save vectors and keys
    const auto& vectors = db.getAllVectors();
    size_t vectorCount = vectors.size();
    file.write(reinterpret_cast<const char*>(&vectorCount), sizeof(vectorCount));

    size_t num_vectors_written = 0;
    for (const auto& [key, vector] : vectors) {
        size_t keySize = key.size();
        file.write(reinterpret_cast<const char*>(&keySize), sizeof(keySize));
        file.write(key.c_str(), keySize);
        file.write(reinterpret_cast<const char*>(vector.data_ptr()), dimensions * sizeof(float));

        // Print the first 5 vectors being saved
        if (num_vectors_written < 5) {
            std::cout << "Saving " << key << ": ";
            for (size_t i = 0; i < std::min(dimensions, size_t(5)); ++i) {
                std::cout << vector[i] << " ";
            }
            std::cout << "..." << std::endl;
        }
        num_vectors_written++;
    }
}

VectorDatabase Persistence::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Load database metadata
    size_t dimensions;
    bool useApproximate;
    file.read(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
    file.read(reinterpret_cast<char*>(&useApproximate), sizeof(useApproximate));

    VectorDatabase db(dimensions, useApproximate);

    // Load vectors and keys
    size_t vectorCount;
    file.read(reinterpret_cast<char*>(&vectorCount), sizeof(vectorCount));

    std::vector<float> buffer(dimensions);
    std::string key;
    for (size_t i = 0; i < vectorCount; ++i) {
        size_t keySize;
        file.read(reinterpret_cast<char*>(&keySize), sizeof(keySize));

        key.resize(keySize);
        file.read(&key[0], keySize);

        file.read(reinterpret_cast<char*>(buffer.data()), dimensions * sizeof(float));

        Vector vector = Vector::read_from(file, dimensions);
        db.insert(vector, key);

        // Print the first 5 vectors being loaded
        if (i < 5) {
            std::cout << "Loading " << key << ": ";
            for (size_t j = 0; j < std::min(dimensions, size_t(5)); ++j) {
                std::cout << vector[j] << " ";
            }
            std::cout << "..." << std::endl;
        }
    }

    return db;
}