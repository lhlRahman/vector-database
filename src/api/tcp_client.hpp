#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "protocol.hpp"

class TCPClient {
public:
    struct SearchResult {
        std::string key;
        float distance;
    };

    struct BatchResult {
        uint32_t successful;
        uint32_t failed;
    };

    struct Stats {
        uint32_t total_vectors;
        uint32_t total_searches;
        uint32_t total_inserts;
        uint32_t total_deletes;
    };

    TCPClient(const std::string& host = "127.0.0.1", int port = 9090);
    ~TCPClient() noexcept;

    TCPClient(const TCPClient&) = delete;
    TCPClient& operator=(const TCPClient&) = delete;

    void connect();
    void disconnect();
    bool is_connected() const { return fd_ >= 0; }

    // Core operations
    bool ping();
    bool insert(const std::string& key, const float* data, size_t dims, const std::string& metadata = "");
    bool insert(const std::string& key, const std::vector<float>& data, const std::string& metadata = "");
    std::vector<SearchResult> search(const float* data, size_t dims, size_t k);
    std::vector<SearchResult> search(const std::vector<float>& data, size_t k);
    bool remove(const std::string& key);
    bool update(const std::string& key, const float* data, size_t dims, const std::string& metadata = "");
    bool update(const std::string& key, const std::vector<float>& data, const std::string& metadata = "");

    struct GetResult {
        std::vector<float> vector;
        std::string metadata;
    };
    std::optional<GetResult> get(const std::string& key);

    // Batch operations
    BatchResult batch_insert(const std::vector<std::string>& keys,
                             const std::vector<std::vector<float>>& vectors,
                             const std::vector<std::string>& metadata = {});
    BatchResult batch_delete(const std::vector<std::string>& keys);

    // Configuration
    bool set_algorithm(const std::string& algo);
    bool set_metric(const std::string& metric);

    // Stats
    Stats stats();

private:
    std::string host_;
    int port_;
    int fd_ = -1;

    // Send a request and receive the response
    struct Response {
        proto::Status status;
        std::vector<uint8_t> payload;
    };

    void send_request(proto::Command cmd, const std::vector<uint8_t>& payload);
    Response recv_response();
    Response request(proto::Command cmd, const std::vector<uint8_t>& payload = {});

    static bool recv_exact(int fd, void* buf, size_t n);
    static bool send_all(int fd, const void* buf, size_t n);
};
