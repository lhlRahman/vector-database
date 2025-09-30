
#include <cstring>
#include <iostream>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>

#include "vector_db_server.hpp"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --dimensions <n>     Vector dimensions (default: 128)\n";
    std::cout << "  --host <host>        Server host (default: localhost)\n";
    std::cout << "  --port <port>        Server port (default: 8080)\n";
    std::cout << "  --db-file <file>     Database file path (default: vectors.db)\n";
    std::cout << "  --disable-recovery   Disable recovery endpoints\n";
    std::cout << "  --disable-batch      Disable batch operation endpoints\n";
    std::cout << "  --disable-stats      Disable statistics endpoints\n";
    std::cout << "  --no-persistence     Disable atomic persistence (for testing)\n";
    std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default configuration
    size_t dimensions = 128;
    std::string host = "localhost";
    int port = 8080;
    std::string db_file = "vectors.db";
    bool enable_recovery = true;
    bool enable_batch = true;
    bool enable_stats = true;
    bool enable_persistence = true;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--dimensions" && i + 1 < argc) {
            dimensions = std::stoul(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--db-file" && i + 1 < argc) {
            db_file = argv[++i];
        } else if (arg == "--disable-recovery") {
            enable_recovery = false;
        } else if (arg == "--disable-batch") {
            enable_batch = false;
        } else if (arg == "--disable-stats") {
            enable_stats = false;
        } else if (arg == "--no-persistence") {
            enable_persistence = false;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    try {
        std::cout << "Vector Database Server" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "Dimensions: " << dimensions << std::endl;
        std::cout << "Host: " << host << std::endl;
        std::cout << "Port: " << port << std::endl;
        std::cout << "Database file: " << db_file << std::endl;
        std::cout << "Atomic persistence: " << (enable_persistence ? "enabled" : "disabled") << std::endl;
        std::cout << "Recovery endpoints: " << (enable_recovery ? "enabled" : "disabled") << std::endl;
        std::cout << "Batch endpoints: " << (enable_batch ? "enabled" : "disabled") << std::endl;
        std::cout << "Statistics endpoints: " << (enable_stats ? "enabled" : "disabled") << std::endl;
        std::cout << std::endl;
        
        std::cout << "Starting server..." << std::endl;
        
        // Create necessary directories if persistence is enabled
        if (enable_persistence) {
            try {
                std::filesystem::create_directories("data");
                std::cout << "Created data directories for persistence" << std::endl;
            } catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Warning: Could not create data directories: " << e.what() << std::endl;
                std::cout << "Continuing without persistence..." << std::endl;
                enable_persistence = false;
            }
        }
        
        // Create and start server with VectorDBServer's existing constructor
        // The VectorDBServer constructor takes: dimensions, db_file, host, port, and the enable flags
        VectorDBServer server(dimensions, db_file, host, port, 
                            enable_recovery, enable_batch, enable_stats);
        // Start the server in blocking mode
        server.start(true);

        
        std::cout << "Server shutdown completed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal: Unknown error occurred" << std::endl;
        return 1;
    }
    
    return 0;
}