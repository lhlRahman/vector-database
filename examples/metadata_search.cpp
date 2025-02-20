#include "../include/vector_database.hpp"
#include "../src/utils/random_generator.hpp"
#include <iostream>
#include <string>
#include <vector>

Vector create_text_embedding(const std::string& text) {
    // This is a placeholder function - in reality, you would:
    // 1. Use a text embedding model (like BERT/GPT/etc.)
    // 2. Convert text to embedding vector
    RandomGenerator rng;
    return rng.generateNormalVector(384, 0.0f, 0.1f);  // 384-dimensional vector
}

int main() {
    // Create a vector database with 384 dimensions
    // (384 is common for text embeddings, but depends on your model)
    VectorDatabase db(384);

    // Create embeddings for two similar texts
    std::string text1 = "The quick brown fox jumps over the lazy dog";
    std::string text2 = "A quick brown dog jumps over the lazy fox";
    
    Vector embedding1 = create_text_embedding(text1);
    Vector embedding2 = create_text_embedding(text2);

    // Insert vectors with their original text as metadata
    db.insert(embedding1, "text1", text1);
    db.insert(embedding2, "text2", text2);

    // Create a query embedding
    std::string query_text = "brown fox jumping";
    Vector query_embedding = create_text_embedding(query_text);

    // Search for 5 most similar texts
    auto results = db.similaritySearchWithMetadata(query_embedding, 5);

    // Print results
    std::cout << "Query: " << query_text << "\n\n";
    for (const auto& result : results) {
        std::cout << "Distance: " << result.distance << "\n";
        std::cout << "Document: " << result.metadata << "\n";
        std::cout << "Key: " << result.key << "\n\n";
    }

    // Save database for later use
    db.saveToFile("text_vectors.db");

    // Later, you can load the database:
    VectorDatabase loaded_db(384);
    loaded_db.loadFromFile("text_vectors.db");

    return 0;
}