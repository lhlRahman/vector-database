#include "../include/vector_database.hpp"
#include "../src/utils/random_generator.hpp"
#include "../src/features/dimensionality_reduction.hpp"
#include "../src/features/persistence.hpp"
#include "../src/features/query_cache.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>

int main() {
    try {
        // Create a vector database for 128-dimensional vectors
        VectorDatabase db(128);

        // Create a random generator
        RandomGenerator rng;

        // Insert 1000 random vectors
        for (int i = 0; i < 20; ++i) {
            Vector v = rng.generateUniformVector(128);
            try {
                db.insert(v, "vector_" + std::to_string(i));
            } catch (const std::exception& e) {
                std::cerr << "Error inserting vector " << i << ": " << e.what() << std::endl;
            }
        }

        std::cout << "Inserted " << db.getAllVectors().size() << " vectors into the original database." << std::endl;

        // Use PCA to reduce dimensionality
        PCA pca(64);  // Reduce to 64 dimensions
        const auto& vectorMap = db.getAllVectors();
        std::vector<Vector> vectors;
        vectors.reserve(vectorMap.size());
        for (const auto& [key, vector] : vectorMap) {
            vectors.push_back(vector);
        }
        pca.fit(vectors);

        // Transform vectors and create a new database
        VectorDatabase reduced_db(64);
        int successful_transforms = 0;
        for (const auto& [key, vector] : vectorMap) {
            try {
                Vector reduced = pca.transform(vector);
                reduced_db.insert(reduced, key);
                successful_transforms++;
            } catch (const std::exception& e) {
                std::cerr << "Error transforming and inserting vector " << key << ": " << e.what() << std::endl;
            }
        }

        std::cout << "Successfully transformed and inserted " << successful_transforms << " vectors into the reduced database." << std::endl;

        // Save the reduced database
        // Persistence::save(reduced_db, "reduced_db.bin");

        // Load the database
        // VectorDatabase loaded_db = Persistence::load("reduced_db.bin");

        std::cout << "Loaded " << reduced_db.getAllVectors().size() << " vectors from the saved database." << std::endl;

        // Use query cache
        QueryCache cache(100);  // Cache size of 100

        // Create a fixed set of query vectors
        std::vector<Vector> fixed_queries;
        for (int i = 0; i < 10; ++i) {
            fixed_queries.push_back(rng.generateUniformVector(64));
        }

        // Perform similarity searches
        for (int i = 0; i < 20; ++i) {
            Vector query = fixed_queries[i % 10];  // Cycle through the fixed queries
            std::vector<std::pair<std::string, float>> results;

            if (!cache.get(query, results)) {
                try {
                    results = reduced_db.similaritySearch(query, 5);
                    cache.put(query, results);
                    std::cout << "Cache miss" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error in similarity search: " << e.what() << std::endl;
                    continue;
                }
            } else {
                std::cout << "Cache hit" << std::endl;
            }

            std::cout << "Query " << i << " results:" << std::endl;
            for (const auto& [key, distance] : results) {
                std::cout << key << ": distance = " << distance << std::endl;
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
