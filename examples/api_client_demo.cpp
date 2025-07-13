// examples/api_client_demo.cpp
#include "../cpp-httplib/httplib.h"
#include "../json.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using json = nlohmann::json;

// Helper function to generate random vectors
std::vector<float> generateRandomVector(size_t dimensions) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    std::vector<float> vec(dimensions);
    for (auto& v : vec) {
        v = dis(gen);
    }
    
    // Normalize the vector
    float norm = 0.0f;
    for (auto v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    for (auto& v : vec) {
        v /= norm;
    }
    
    return vec;
}

int main() {
    httplib::Client client("localhost", 8080);
    client.set_connection_timeout(30);
    
    std::cout << "=== Vector Database API Client Demo ===" << std::endl;
    
    // 1. Check server health
    std::cout << "\n1. Checking server health..." << std::endl;
    auto health_res = client.Get("/health");
    if (health_res && health_res->status == 200) {
        auto health = json::parse(health_res->body);
        std::cout << "   Server status: " << health["status"] << std::endl;
        std::cout << "   Service: " << health["service"] << std::endl;
        std::cout << "   Version: " << health["version"] << std::endl;
    } else {
        std::cout << "   Failed to connect to server!" << std::endl;
        return 1;
    }
    
    // 2. Get database info
    std::cout << "\n2. Getting database info..." << std::endl;
    auto info_res = client.Get("/info");
    if (info_res && info_res->status == 200) {
        auto info = json::parse(info_res->body);
        std::cout << "   Dimensions: " << info["dimensions"] << std::endl;
        std::cout << "   Approximate search: " << info["use_approximate"] << std::endl;
        std::cout << "   Vector count: " << info["vector_count"] << std::endl;
    }
    
    // 3. Insert single vector
    std::cout << "\n3. Inserting a single vector..." << std::endl;
    json insert_request = {
        {"key", "test_vector_1"},
        {"vector", generateRandomVector(128)},
        {"metadata", "This is a test vector"}
    };
    
    auto insert_res = client.Post("/vectors", insert_request.dump(), "application/json");
    if (insert_res && insert_res->status == 200) {
        auto result = json::parse(insert_res->body);
        std::cout << "   Status: " << result["status"] << std::endl;
        std::cout << "   Key: " << result["key"] << std::endl;
    }
    
    // 4. Batch insert vectors
    std::cout << "\n4. Batch inserting 5 vectors..." << std::endl;
    json batch_request = {{"vectors", json::array()}};
    
    for (int i = 2; i <= 6; ++i) {
        json vec_obj = {
            {"key", "test_vector_" + std::to_string(i)},
            {"vector", generateRandomVector(128)}
        };
        batch_request["vectors"].push_back(vec_obj);
    }
    
    auto batch_res = client.Post("/vectors/batch", batch_request.dump(), "application/json");
    if (batch_res && batch_res->status == 200) {
        auto result = json::parse(batch_res->body);
        std::cout << "   Status: " << result["status"] << std::endl;
        std::cout << "   Count: " << result["count"] << std::endl;
    }
    
    // 5. Search for similar vectors
    std::cout << "\n5. Searching for similar vectors..." << std::endl;
    auto query_vector = generateRandomVector(128);
    json search_request = {
        {"vector", query_vector},
        {"k", 3},
        {"with_metadata", true}
    };
    
    auto search_res = client.Post("/search", search_request.dump(), "application/json");
    if (search_res && search_res->status == 200) {
        auto result = json::parse(search_res->body);
        std::cout << "   Found " << result["count"] << " similar vectors:" << std::endl;
        
        for (const auto& res : result["results"]) {
            std::cout << "   - Key: " << res["key"] 
                     << ", Distance: " << res["distance"];
            if (res.contains("metadata") && !res["metadata"].is_null()) {
                std::cout << ", Metadata: " << res["metadata"];
            }
            std::cout << std::endl;
        }
    }
    
    // 6. Toggle approximate search
    std::cout << "\n6. Enabling approximate search..." << std::endl;
    json approx_request = {{"enabled", true}};
    
    auto approx_res = client.Put("/config/approximate", approx_request.dump(), "application/json");
    if (approx_res && approx_res->status == 200) {
        auto result = json::parse(approx_res->body);
        std::cout << "   Status: " << result["status"] << std::endl;
        std::cout << "   Approximate search: " << result["approximate_search"] << std::endl;
    }
    
    // 7. Search again with approximate search
    std::cout << "\n7. Searching with approximate search enabled..." << std::endl;
    auto search_approx_res = client.Post("/search", search_request.dump(), "application/json");
    if (search_approx_res && search_approx_res->status == 200) {
        auto result = json::parse(search_approx_res->body);
        std::cout << "   Found " << result["count"] << " similar vectors (LSH):" << std::endl;
        
        for (const auto& res : result["results"]) {
            std::cout << "   - Key: " << res["key"] 
                     << ", Distance: " << res["distance"] << std::endl;
        }
    }
    
    // 8. Get specific vector
    std::cout << "\n8. Getting specific vector..." << std::endl;
    auto get_vec_res = client.Get("/vectors/test_vector_1");
    if (get_vec_res && get_vec_res->status == 200) {
        auto result = json::parse(get_vec_res->body);
        std::cout << "   Key: " << result["key"] << std::endl;
        std::cout << "   Dimensions: " << result["vector"].size() << std::endl;
        if (result.contains("metadata")) {
            std::cout << "   Metadata: " << result["metadata"] << std::endl;
        }
    }
    
    // 9. List all vectors (paginated)
    std::cout << "\n9. Listing all vectors..." << std::endl;
    auto list_res = client.Get("/vectors?page=1&per_page=10");
    if (list_res && list_res->status == 200) {
        auto result = json::parse(list_res->body);
        std::cout << "   Total vectors: " << result["total"] << std::endl;
        std::cout << "   Page: " << result["page"] << "/" << result["total_pages"] << std::endl;
        std::cout << "   Vectors on this page:" << std::endl;
        
        for (const auto& vec : result["vectors"]) {
            std::cout << "   - " << vec["key"] << std::endl;
        }
    }
    
    // 10. Save database
    std::cout << "\n10. Saving database to disk..." << std::endl;
    auto save_res = client.Post("/save", "", "application/json");
    if (save_res && save_res->status == 200) {
        auto result = json::parse(save_res->body);
        std::cout << "   Status: " << result["status"] << std::endl;
        std::cout << "   File: " << result["file"] << std::endl;
    }
    
    std::cout << "\n=== Demo completed successfully! ===" << std::endl;
    
    return 0;
}