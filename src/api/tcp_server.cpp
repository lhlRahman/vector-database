#include "tcp_server.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

TCPServer::TCPServer(size_t dimensions,
                     const std::string& host,
                     int port,
                     size_t thread_pool_size)
    : host_(host), port_(port), pool_size_(thread_pool_size) {
    db_ = std::make_unique<VectorDatabase>(dimensions, "hnsw",
                                           /*enable_atomic_persistence=*/false,
                                           /*enable_batch_operations=*/true);
    db_->initialize();
}

TCPServer::~TCPServer() noexcept {
    stop();
}

void TCPServer::start() {
    // Create listen socket
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        throw std::runtime_error("socket(): " + std::string(strerror(errno)));
    }

    int opt = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);

    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(listen_fd_);
        throw std::runtime_error("bind(): " + std::string(strerror(errno)));
    }

    if (listen(listen_fd_, 128) < 0) {
        close(listen_fd_);
        throw std::runtime_error("listen(): " + std::string(strerror(errno)));
    }

    // Pipe for signaling workers about new connections
    if (pipe(pipe_fd_) < 0) {
        close(listen_fd_);
        throw std::runtime_error("pipe(): " + std::string(strerror(errno)));
    }

    running_.store(true, std::memory_order_release);

    // Spawn worker threads
    for (size_t i = 0; i < pool_size_; ++i) {
        workers_.emplace_back(&TCPServer::worker_loop, this);
    }

    // Accept thread
    accept_thread_ = std::thread(&TCPServer::accept_loop, this);

    std::cout << "TCP server listening on " << host_ << ":" << port_ << "\n";
}

void TCPServer::stop() {
    if (!running_.exchange(false)) return;

    // Close listen socket to unblock accept()
    if (listen_fd_ >= 0) {
        shutdown(listen_fd_, SHUT_RDWR);
        close(listen_fd_);
        listen_fd_ = -1;
    }

    // Signal workers to wake up and exit — send -1 for each worker
    for (size_t i = 0; i < pool_size_; ++i) {
        int sentinel = -1;
        (void)write(pipe_fd_[1], &sentinel, sizeof(sentinel));
    }

    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();

    if (pipe_fd_[0] >= 0) { close(pipe_fd_[0]); pipe_fd_[0] = -1; }
    if (pipe_fd_[1] >= 0) { close(pipe_fd_[1]); pipe_fd_[1] = -1; }
}

void TCPServer::accept_loop() {
    int consecutive_errors = 0;
    while (running_.load(std::memory_order_acquire)) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_.load(std::memory_order_acquire)) break;
            // Backoff on repeated accept failures (e.g. fd limit)
            if (++consecutive_errors > 10) {
                std::cerr << "[server] accept() failing repeatedly: "
                          << strerror(errno) << '\n';
                usleep(100'000); // 100ms backoff
            }
            continue;
        }
        consecutive_errors = 0;

        // Disable Nagle for low latency
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        // Set recv timeout (30s) so stalled clients don't block workers forever
        struct timeval tv{30, 0};
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        // Pass fd to a worker via pipe
        ssize_t w = write(pipe_fd_[1], &client_fd, sizeof(client_fd));
        if (w != sizeof(client_fd)) {
            std::cerr << "[server] pipe write failed, dropping connection\n";
            close(client_fd);
        }
    }
}

void TCPServer::worker_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        // Poll on the pipe read end
        pollfd pfd{};
        pfd.fd = pipe_fd_[0];
        pfd.events = POLLIN;

        int ret = poll(&pfd, 1, 100); // 100ms timeout to check running_
        if (ret <= 0) continue;

        int client_fd = -1;
        ssize_t r = read(pipe_fd_[0], &client_fd, sizeof(client_fd));
        if (r != sizeof(client_fd) || client_fd < 0) {
            continue; // sentinel or error
        }

        handle_connection(client_fd);
    }
}

void TCPServer::handle_connection(int client_fd) {
    // Keep-alive: handle multiple requests on same connection
    while (running_.load(std::memory_order_relaxed)) {
        // Read header
        uint8_t header[proto::HEADER_SIZE];
        if (!recv_exact(client_fd, header, proto::HEADER_SIZE)) break;

        // Parse header
        uint16_t magic;
        std::memcpy(&magic, header, 2);
        if (magic != proto::MAGIC) {
            send_error(client_fd, "invalid magic bytes");
            break;
        }

        auto cmd = static_cast<proto::Command>(header[2]);
        uint32_t payload_len;
        std::memcpy(&payload_len, header + 3, 4);

        if (payload_len > proto::MAX_PAYLOAD_SIZE) {
            send_error(client_fd, "payload too large");
            break;
        }

        // Read payload
        std::vector<uint8_t> payload(payload_len);
        if (payload_len > 0 && !recv_exact(client_fd, payload.data(), payload_len)) {
            break;
        }

        dispatch(client_fd, cmd, payload.data(), payload_len);
    }

    close(client_fd);
}

bool TCPServer::recv_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t r = recv(fd, p, remaining, 0);
        if (r <= 0) return false;
        p += r;
        remaining -= static_cast<size_t>(r);
    }
    return true;
}

bool TCPServer::send_all(int fd, const void* buf, size_t n) {
    auto* p = static_cast<const uint8_t*>(buf);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t w = send(fd, p, remaining, 0);
        if (w <= 0) return false;
        p += w;
        remaining -= static_cast<size_t>(w);
    }
    return true;
}

void TCPServer::send_response(int fd, proto::Status status, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> frame;
    frame.reserve(proto::HEADER_SIZE + payload.size());
    proto::write_response_header(frame, status, static_cast<uint32_t>(payload.size()));
    frame.insert(frame.end(), payload.begin(), payload.end());
    send_all(fd, frame.data(), frame.size());
}

void TCPServer::send_error(int fd, const std::string& msg) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_string(msg);
    send_response(fd, proto::Status::ERROR, payload);
}

void TCPServer::send_ok(int fd, const std::vector<uint8_t>& payload) {
    send_response(fd, proto::Status::OK, payload);
}

// ── Dispatch ───────────────────────────────────────────────────

void TCPServer::dispatch(int fd, proto::Command cmd, const uint8_t* payload, size_t len) {
    try {
        switch (cmd) {
            case proto::Command::PING:         handle_ping(fd); break;
            case proto::Command::INSERT:       handle_insert(fd, payload, len); break;
            case proto::Command::SEARCH:       handle_search(fd, payload, len); break;
            case proto::Command::DELETE:       handle_delete(fd, payload, len); break;
            case proto::Command::UPDATE:       handle_update(fd, payload, len); break;
            case proto::Command::GET:          handle_get(fd, payload, len); break;
            case proto::Command::BATCH_INSERT: handle_batch_insert(fd, payload, len); break;
            case proto::Command::BATCH_DELETE: handle_batch_delete(fd, payload, len); break;
            case proto::Command::SET_ALGORITHM:handle_set_algorithm(fd, payload, len); break;
            case proto::Command::SET_METRIC:   handle_set_metric(fd, payload, len); break;
            case proto::Command::STATS:        handle_stats(fd); break;
            default:
                send_error(fd, "unknown command");
                break;
        }
    } catch (const std::exception& e) {
        send_error(fd, e.what());
    }
}

// ── Command handlers ───────────────────────────────────────────

void TCPServer::handle_ping(int fd) {
    send_ok(fd);
}

void TCPServer::handle_insert(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    std::string key = r.read_string();
    uint32_t dims = r.read_u32();
    const float* data = r.read_floats_ptr(dims);
    std::string metadata = r.read_string();

    Vector v(std::vector<float>(data, data + dims));
    bool ok = db_->insert(v, key, metadata);
    if (ok) {
        send_ok(fd);
    } else {
        send_error(fd, "insert failed: key may already exist");
    }
}

void TCPServer::handle_search(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    uint32_t dims = r.read_u32();
    const float* data = r.read_floats_ptr(dims);
    uint32_t k = r.read_u32();

    Vector query(std::vector<float>(data, data + dims));
    auto results = db_->similaritySearch(query, k);

    std::vector<uint8_t> resp;
    proto::BufferWriter w(resp);
    w.write_u32(static_cast<uint32_t>(results.size()));
    for (const auto& [key, dist] : results) {
        w.write_string(key);
        w.write_f32(dist);
    }
    send_ok(fd, resp);
}

void TCPServer::handle_delete(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    std::string key = r.read_string();

    bool ok = db_->remove(key);
    if (ok) {
        send_ok(fd);
    } else {
        send_response(fd, proto::Status::NOT_FOUND, {});
    }
}

void TCPServer::handle_update(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    std::string key = r.read_string();
    uint32_t dims = r.read_u32();
    const float* data = r.read_floats_ptr(dims);
    std::string metadata = r.read_string();

    Vector v(std::vector<float>(data, data + dims));
    bool ok = db_->update(v, key, metadata);
    if (ok) {
        send_ok(fd);
    } else {
        send_response(fd, proto::Status::NOT_FOUND, {});
    }
}

void TCPServer::handle_get(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    std::string key = r.read_string();

    auto vec = db_->get(key);
    if (!vec) {
        send_response(fd, proto::Status::NOT_FOUND, {});
        return;
    }

    std::string metadata = db_->getMetadata(key);

    std::vector<uint8_t> resp;
    proto::BufferWriter w(resp);
    w.write_u32(static_cast<uint32_t>(vec->size()));
    w.write_floats(vec->data_ptr(), vec->size());
    w.write_string(metadata);
    send_ok(fd, resp);
}

void TCPServer::handle_batch_insert(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    uint32_t count = r.read_u32();

    std::vector<std::string> keys;
    std::vector<Vector> vectors;
    std::vector<std::string> metadata;
    keys.reserve(count);
    vectors.reserve(count);
    metadata.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        keys.push_back(r.read_string());
        uint32_t dims = r.read_u32();
        const float* data = r.read_floats_ptr(dims);
        vectors.emplace_back(std::vector<float>(data, data + dims));
        metadata.push_back(r.read_string());
    }

    auto result = db_->batchInsert(keys, vectors, metadata);

    std::vector<uint8_t> resp;
    proto::BufferWriter w(resp);
    w.write_u32(static_cast<uint32_t>(result.operations_committed));
    w.write_u32(static_cast<uint32_t>(count - result.operations_committed));
    send_ok(fd, resp);
}

void TCPServer::handle_batch_delete(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    uint32_t count = r.read_u32();

    std::vector<std::string> keys;
    keys.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        keys.push_back(r.read_string());
    }

    auto result = db_->batchDelete(keys);

    std::vector<uint8_t> resp;
    proto::BufferWriter w(resp);
    w.write_u32(static_cast<uint32_t>(result.operations_committed));
    w.write_u32(static_cast<uint32_t>(count - result.operations_committed));
    send_ok(fd, resp);
}

void TCPServer::handle_set_algorithm(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    std::string algo = r.read_string();

    if (algo == "exact") {
        db_->setApproximateAlgorithm("exact", 0, 0);
    } else if (algo == "lsh") {
        db_->setApproximateAlgorithm("lsh", 10, 8);
    } else if (algo == "hnsw") {
        db_->setApproximateAlgorithm("hnsw", 10, 8);
    } else {
        send_error(fd, "unknown algorithm: " + algo);
        return;
    }
    send_ok(fd);
}

void TCPServer::handle_set_metric(int fd, const uint8_t* payload, size_t len) {
    proto::BufferReader r(payload, len);
    std::string metric = r.read_string();

    if (metric == "euclidean") {
        db_->setDistanceMetric(std::make_shared<EuclideanDistance>());
    } else if (metric == "manhattan") {
        db_->setDistanceMetric(std::make_shared<ManhattanDistance>());
    } else if (metric == "cosine") {
        db_->setDistanceMetric(std::make_shared<CosineSimilarity>());
    } else {
        send_error(fd, "unknown metric: " + metric);
        return;
    }
    send_ok(fd);
}

void TCPServer::handle_stats(int fd) {
    auto stats = db_->getStatistics();

    std::vector<uint8_t> resp;
    proto::BufferWriter w(resp);
    w.write_u32(static_cast<uint32_t>(stats.total_vectors));
    w.write_u32(static_cast<uint32_t>(stats.total_searches));
    w.write_u32(static_cast<uint32_t>(stats.total_inserts));
    w.write_u32(static_cast<uint32_t>(stats.total_deletes));
    send_ok(fd, resp);
}
