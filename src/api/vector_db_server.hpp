
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "../../cpp-httplib/httplib.h"
#include "json.hpp"
#include "../core/vector_database.hpp"

/**
 * Vector Database Server
 * 
 * RESTful API server with atomic operations, recovery capabilities,
 * and batch operations support.
 */
class VectorDBServer {
private:
    std::unique_ptr<VectorDatabase> db;
    std::mutex db_mutex;
    httplib::Server server;
    size_t dimensions;
    std::string db_file;
    
    // Server configuration
    std::string host;
    int port;
    bool enable_recovery_endpoints;
    bool enable_batch_endpoints;
    bool enable_statistics_endpoints;
    
    // Statistics
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> total_batch_operations{0};
    
    // Recovery monitoring
    std::atomic<bool> recovery_in_progress{false};
    std::thread recovery_monitor_thread;
    std::atomic<bool> should_stop_monitoring{false};
    
    // Current distance metric
    std::string current_distance_metric{"euclidean"};
    
    // Current algorithm
    std::string current_algorithm{"hnsw"};
    
    // Setup routes
    void setupRoutes();
    void setupHealthRoutes();
    void setupVectorRoutes();
    void setupBatchRoutes();
    void setupRecoveryRoutes();
    void setupStatisticsRoutes();
    void setupConfigRoutes();
    
    // Route handlers
    void handleHealth(const httplib::Request& req, httplib::Response& res);
    void handleGetVectors(const httplib::Request& req, httplib::Response& res);
    void handlePostVectors(const httplib::Request& req, httplib::Response& res);
    void handleGetVector(const httplib::Request& req, httplib::Response& res);
    void handlePutVector(const httplib::Request& req, httplib::Response& res);
    void handleDeleteVector(const httplib::Request& req, httplib::Response& res);
    void handleSearch(const httplib::Request& req, httplib::Response& res);
    void handleBatchSearch(const httplib::Request& req, httplib::Response& res);
    
    // Batch operation handlers
    void handleBatchInsert(const httplib::Request& req, httplib::Response& res);
    void handleBatchUpdate(const httplib::Request& req, httplib::Response& res);
    void handleBatchDelete(const httplib::Request& req, httplib::Response& res);
    void handleMixedBatch(const httplib::Request& req, httplib::Response& res);
    
    // Recovery handlers
    void handleRecoveryStatus(const httplib::Request& req, httplib::Response& res);
    void handleRecoveryInfo(const httplib::Request& req, httplib::Response& res);
    void handleForceCheckpoint(const httplib::Request& req, httplib::Response& res);
    void handleForceFlush(const httplib::Request& req, httplib::Response& res);
    
    // Statistics handlers
    void handleStatistics(const httplib::Request& req, httplib::Response& res);
    void handleDatabaseStats(const httplib::Request& req, httplib::Response& res);
    void handlePersistenceStats(const httplib::Request& req, httplib::Response& res);
    void handleBatchStats(const httplib::Request& req, httplib::Response& res);
    
    // Configuration handlers
    void handleGetConfig(const httplib::Request& req, httplib::Response& res);
    void handleUpdateConfig(const httplib::Request& req, httplib::Response& res);
    void handleUpdatePersistenceConfig(const httplib::Request& req, httplib::Response& res);
    
    // Utility functions
    void logRequest(const std::string& method, const std::string& path, int status_code);
    void handleError(httplib::Response& res, int status_code, const std::string& message);
    void handleSuccess(httplib::Response& res, const nlohmann::json& data = nlohmann::json{});
    
    // Recovery monitoring
    void recoveryMonitorFunction();
    void startRecoveryMonitoring();
    void stopRecoveryMonitoring();
    
    // Validation
    bool validateVector(const nlohmann::json& vector_json, size_t expected_dimensions);
    bool validateBatchRequest(const nlohmann::json& request_json);
    
public:
    /**
     * Constructor
     * @param dims Vector dimensions
     * @param dbFile Database file path
     * @param host Server host
     * @param port Server port
     * @param enable_recovery Enable recovery endpoints
     * @param enable_batch Enable batch operation endpoints
     * @param enable_stats Enable statistics endpoints
     */
    VectorDBServer(size_t dims, 
                          const std::string& dbFile = "vectors.db",
                          const std::string& host = "localhost",
                          int port = 8080,
                          bool enable_recovery = true,
                          bool enable_batch = true,
                          bool enable_stats = true);
    
    /**
     * Destructor
     */
    ~VectorDBServer();
    
    // Non-copyable
    VectorDBServer(const VectorDBServer&) = delete;
    VectorDBServer& operator=(const VectorDBServer&) = delete;
    
    /**
     * Start the server
     * @param blocking Whether to block the calling thread
     */
    void start(bool blocking = true);
    
    /**
     * Stop the server
     */
    void stop();
    
    /**
     * Check if server is running
     * @return true if running
     */
    bool isRunning() const;
    
    /**
     * Get server statistics
     */
    struct ServerStatistics {
        uint64_t total_requests;
        uint64_t successful_requests;
        uint64_t failed_requests;
        uint64_t total_batch_operations;
        bool recovery_in_progress;
        bool server_running;
        std::string host;
        int port;
    };
    
    ServerStatistics getServerStatistics() const;
    
    /**
     * Update server configuration
     * @param new_host New host
     * @param new_port New port
     * @param enable_recovery Enable recovery endpoints
     * @param enable_batch Enable batch endpoints
     * @param enable_stats Enable statistics endpoints
     */
    void updateConfig(const std::string& new_host,
                     int new_port,
                     bool enable_recovery,
                     bool enable_batch,
                     bool enable_stats);
    
    // SIMD configuration handlers
    void handleToggleSIMD(const httplib::Request& req, httplib::Response& res);
    void handleGetSIMDStatus(const httplib::Request& req, httplib::Response& res);
    
    // Distance metric configuration handlers
    void handleSetDistanceMetric(const httplib::Request& req, httplib::Response& res);
    void handleGetDistanceMetric(const httplib::Request& req, httplib::Response& res);
    
    // Algorithm configuration handlers
    void handleSetAlgorithm(const httplib::Request& req, httplib::Response& res);
    void handleGetAlgorithm(const httplib::Request& req, httplib::Response& res);
};
