#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../core/vector_database.hpp"
#include "protocol.hpp"

class TCPServer {
public:
    TCPServer(size_t dimensions,
              const std::string& host = "0.0.0.0",
              int port = 9090,
              size_t thread_pool_size = 4);

    ~TCPServer() noexcept;

    TCPServer(const TCPServer&) = delete;
    TCPServer& operator=(const TCPServer&) = delete;

    void start();
    void stop();
    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    VectorDatabase& database() { return *db_; }

private:
    std::unique_ptr<VectorDatabase> db_;
    std::string host_;
    int port_;
    size_t pool_size_;

    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::vector<std::thread> workers_;

    // Connection queue (lock-free-ish via pipe)
    int pipe_fd_[2] = {-1, -1};

    void accept_loop();
    void worker_loop();
    void handle_connection(int client_fd);

    // Read exactly n bytes from fd, returns false on disconnect/error
    static bool recv_exact(int fd, void* buf, size_t n);
    static bool send_all(int fd, const void* buf, size_t n);

    // Protocol dispatch
    void dispatch(int fd, proto::Command cmd, const uint8_t* payload, size_t len);

    // Command handlers — each reads from payload, writes response to fd
    void handle_ping(int fd);
    void handle_insert(int fd, const uint8_t* payload, size_t len);
    void handle_search(int fd, const uint8_t* payload, size_t len);
    void handle_delete(int fd, const uint8_t* payload, size_t len);
    void handle_update(int fd, const uint8_t* payload, size_t len);
    void handle_get(int fd, const uint8_t* payload, size_t len);
    void handle_batch_insert(int fd, const uint8_t* payload, size_t len);
    void handle_batch_delete(int fd, const uint8_t* payload, size_t len);
    void handle_set_algorithm(int fd, const uint8_t* payload, size_t len);
    void handle_set_metric(int fd, const uint8_t* payload, size_t len);
    void handle_stats(int fd);

    // Send a response frame
    void send_response(int fd, proto::Status status, const std::vector<uint8_t>& payload);
    void send_error(int fd, const std::string& msg);
    void send_ok(int fd, const std::vector<uint8_t>& payload = {});
};
