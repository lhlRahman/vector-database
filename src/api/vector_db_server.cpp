// src/api/vector_db_server.cpp
#include "../../cpp-httplib/httplib.h"
#include "../../json.hpp"
#include "../../include/vector_database.hpp"
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <chrono>
#include <mutex>

using json = nlohmann::json;

class VectorDBServer {
private:
    std::unique_ptr<VectorDatabase> db;
    std::mutex db_mutex;
    httplib::Server server;
    size_t dimensions;
    std::string db_file;
    
public:
    VectorDBServer(size_t dims, const std::string& dbFile = "api_vectors.db") 
        : dimensions(dims), db_file(dbFile) {
        db = std::make_unique<VectorDatabase>(dimensions);
        
        // Try to load existing database
        try {
            db->loadFromFile(db_file);
            std::cout << "Loaded existing database from " << db_file << std::endl;
        } catch (...) {
            std::cout << "Starting with empty database" << std::endl;
        }
        
        setupRoutes();
    }
    
    void setupRoutes() {
        // Health check endpoint
        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"status", "healthy"},
                {"service", "Vector Database API"},
                {"version", "1.0.0"}
            };
            res.set_content(response.dump(), "application/json");
        });
        
        // Get database info
        server.Get("/info", [this](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> lock(db_mutex);
            json response = {
                {"dimensions", db->getDimensions()},
                {"use_approximate", db->isUsingApproximateSearch()},
                {"vector_count", db->getAllVectors().size()}
            };
            res.set_content(response.dump(), "application/json");
        });
        
        // Store a vector
        server.Post("/vectors", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                // Validate input
                if (!body.contains("key") || !body.contains("vector")) {
                    res.status = 400;
                    json error = {{"error", "Missing required fields: key and vector"}};
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                std::string key = body["key"];
                std::vector<float> vec_data = body["vector"];
                
                if (vec_data.size() != dimensions) {
                    res.status = 400;
                    json error = {
                        {"error", "Dimension mismatch"},
                        {"expected", dimensions},
                        {"received", vec_data.size()}
                    };
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                Vector vector(vec_data);
                std::string metadata = body.value("metadata", "");
                
                {
                    std::lock_guard<std::mutex> lock(db_mutex);
                    if (metadata.empty()) {
                        db->insert(vector, key);
                    } else {
                        db->insert(vector, key, metadata);
                    }
                    db->saveToFile(db_file);
                }
                
                json response = {
                    {"status", "success"},
                    {"key", key},
                    {"dimensions", vec_data.size()}
                };
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Fixed batch insert endpoint in vector_db_server.cpp
        server.Post("/vectors/batch", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                // FIXED: Check if body contains "vectors" array, not if body itself is array
                if (!body.contains("vectors") || !body["vectors"].is_array()) {
                    res.status = 400;
                    json error = {{"error", "Request body must contain a 'vectors' array"}};
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                std::vector<Vector> vectors;
                std::vector<std::string> keys;
                
                for (const auto& item : body["vectors"]) {
                    if (!item.contains("key") || !item.contains("vector")) {
                        res.status = 400;
                        json error = {{"error", "Each item must have 'key' and 'vector' fields"}};
                        res.set_content(error.dump(), "application/json");
                        return;
                    }
                    
                    std::vector<float> vec_data = item["vector"];
                    if (vec_data.size() != dimensions) {
                        res.status = 400;
                        json error = {
                            {"error", "Dimension mismatch"},
                            {"key", item["key"]},
                            {"expected", dimensions},
                            {"received", vec_data.size()}
                        };
                        res.set_content(error.dump(), "application/json");
                        return;
                    }
                    
                    vectors.emplace_back(vec_data);
                    keys.push_back(item["key"]);
                }
                
                {
                    std::lock_guard<std::mutex> lock(db_mutex);
                    db->batchInsert(vectors, keys);
                    db->saveToFile(db_file);
                }
                
                json response = {
                    {"status", "success"},
                    {"count", vectors.size()}
                };
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });
        // Search for similar vectors
        server.Post("/search", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                if (!body.contains("vector")) {
                    res.status = 400;
                    json error = {{"error", "Missing required field: vector"}};
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                std::vector<float> query_data = body["vector"];
                if (query_data.size() != dimensions) {
                    res.status = 400;
                    json error = {
                        {"error", "Dimension mismatch"},
                        {"expected", dimensions},
                        {"received", query_data.size()}
                    };
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                size_t k = body.value("k", 5);
                bool with_metadata = body.value("with_metadata", false);
                
                Vector query(query_data);
                json results_array = json::array();
                
                {
                    std::lock_guard<std::mutex> lock(db_mutex);
                    
                    if (with_metadata) {
                        auto results = db->similaritySearchWithMetadata(query, k);
                        for (const auto& result : results) {
                            results_array.push_back({
                                {"key", result.key},
                                {"distance", result.distance},
                                {"metadata", result.metadata}
                            });
                        }
                    } else {
                        auto results = db->similaritySearch(query, k);
                        for (const auto& [key, distance] : results) {
                            results_array.push_back({
                                {"key", key},
                                {"distance", distance}
                            });
                        }
                    }
                }
                
                json response = {
                    {"results", results_array},
                    {"count", results_array.size()}
                };
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Toggle approximate search
        server.Put("/config/approximate", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                auto body = json::parse(req.body);
                
                if (!body.contains("enabled")) {
                    res.status = 400;
                    json error = {{"error", "Missing required field: enabled"}};
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                bool use_approximate = body["enabled"];
                
                {
                    std::lock_guard<std::mutex> lock(db_mutex);
                    db->toggleApproximateSearch(use_approximate);
                }
                
                json response = {
                    {"status", "success"},
                    {"approximate_search", use_approximate}
                };
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Get all vectors (paginated)
        server.Get("/vectors", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                // Parse pagination parameters
                int page = 1;
                int per_page = 100;
                
                if (req.has_param("page")) {
                    page = std::stoi(req.get_param_value("page"));
                }
                if (req.has_param("per_page")) {
                    per_page = std::min(1000, std::stoi(req.get_param_value("per_page")));
                }
                
                std::lock_guard<std::mutex> lock(db_mutex);
                const auto& all_vectors = db->getAllVectors();
                
                json vectors_array = json::array();
                int start_idx = (page - 1) * per_page;
                int end_idx = std::min(start_idx + per_page, (int)all_vectors.size());
                
                auto it = all_vectors.begin();
                std::advance(it, start_idx);
                
                for (int i = start_idx; i < end_idx; ++i, ++it) {
                    json vector_obj = {
                        {"key", it->first},
                        {"vector", std::vector<float>(it->second.begin(), it->second.end())}
                    };
                    
                    std::string metadata = db->getMetadata(it->first);
                    if (!metadata.empty()) {
                        vector_obj["metadata"] = metadata;
                    }
                    
                    vectors_array.push_back(vector_obj);
                }
                
                json response = {
                    {"vectors", vectors_array},
                    {"page", page},
                    {"per_page", per_page},
                    {"total", all_vectors.size()},
                    {"total_pages", (all_vectors.size() + per_page - 1) / per_page}
                };
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Get a specific vector
        server.Get(R"(/vectors/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                std::string key = req.matches[1];
                
                std::lock_guard<std::mutex> lock(db_mutex);
                const auto& all_vectors = db->getAllVectors();
                
                auto it = all_vectors.find(key);
                if (it == all_vectors.end()) {
                    res.status = 404;
                    json error = {{"error", "Vector not found"}};
                    res.set_content(error.dump(), "application/json");
                    return;
                }
                
                json response = {
                    {"key", key},
                    {"vector", std::vector<float>(it->second.begin(), it->second.end())}
                };
                
                std::string metadata = db->getMetadata(key);
                if (!metadata.empty()) {
                    response["metadata"] = metadata;
                }
                
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });
        
        // Save database
        server.Post("/save", [this](const httplib::Request&, httplib::Response& res) {
            try {
                std::lock_guard<std::mutex> lock(db_mutex);
                db->saveToFile(db_file);
                
                json response = {
                    {"status", "success"},
                    {"file", db_file}
                };
                res.set_content(response.dump(), "application/json");
                
            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });
    }
    
    void start(int port = 8080) {
        std::cout << "Starting Vector Database API Server on port " << port << std::endl;
        std::cout << "Dimensions: " << dimensions << std::endl;
        std::cout << "Database file: " << db_file << std::endl;
        std::cout << "\nAvailable endpoints:" << std::endl;
        std::cout << "  GET  /health" << std::endl;
        std::cout << "  GET  /info" << std::endl;
        std::cout << "  POST /vectors" << std::endl;
        std::cout << "  POST /vectors/batch" << std::endl;
        std::cout << "  POST /search" << std::endl;
        std::cout << "  PUT  /config/approximate" << std::endl;
        std::cout << "  GET  /vectors?page=1&per_page=100" << std::endl;
        std::cout << "  GET  /vectors/{key}" << std::endl;
        std::cout << "  POST /save" << std::endl;
        std::cout << "\nServer is ready!" << std::endl;
        
        server.listen("0.0.0.0", port);
    }
};

int main(int argc, char* argv[]) {
    size_t dimensions = 4096;  // Default dimensions
    int port = 8080;          // Default port
    std::string db_file = "api_vectors.db";  // Default database file
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d" || arg == "--dimensions") {
            if (i + 1 < argc) {
                dimensions = std::stoi(argv[++i]);
            }
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                port = std::stoi(argv[++i]);
            }
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) {
                db_file = argv[++i];
            }
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Vector Database API Server" << std::endl;
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -d, --dimensions <n>  Vector dimensions (default: 128)" << std::endl;
            std::cout << "  -p, --port <port>     Server port (default: 8080)" << std::endl;
            std::cout << "  -f, --file <path>     Database file path (default: api_vectors.db)" << std::endl;
            std::cout << "  -h, --help           Show this help message" << std::endl;
            return 0;
        }
    }
    
    try {
        VectorDBServer server(dimensions, db_file);
        server.start(port);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}