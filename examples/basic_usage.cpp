//examples/basic_usage.cpp
#include "../include/vector_database.hpp"
#include "../src/utils/random_generator.hpp"
#include <iostream>

int main() {
    // Create a vector database for 128-dimensional vectors
    VectorDatabase db(128);

    // Create a random generator
    RandomGenerator rng;

    // Insert 1000 random vectors
    for (int i = 0; i < 1000; ++i) {
        Vector v = rng.generateUniformVector(128);
        db.insert(v, "vector_" + std::to_string(i));
    }

    // Perform a nearest neighbor search
    Vector query = rng.generateUniformVector(128);
    auto results = db.similaritySearch(query, 5);

    std::cout << "Nearest neighbors to the query vector:" << std::endl;
    for (const auto& [key, distance] : results) {
        std::cout << key << ": distance = " << distance << std::endl;
    }

    return 0;
}