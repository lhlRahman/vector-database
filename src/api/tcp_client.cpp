#include "tcp_client.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

TCPClient::TCPClient(const std::string& host, int port)
    : host_(host), port_(port) {}

TCPClient::~TCPClient() noexcept {
    disconnect();
}

void TCPClient::connect() {
    if (fd_ >= 0) return;

    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) {
        throw std::runtime_error("socket(): " + std::string(strerror(errno)));
    }

    // Disable Nagle
    int flag = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    {
        // Resolve host properly (numeric IP or hostname). Previously inet_pton's
        // return was ignored, so a hostname silently dialed 0.0.0.0.
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* res = nullptr;
        int gai = ::getaddrinfo(host_.c_str(), nullptr, &hints, &res);
        if (gai != 0 || res == nullptr) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot resolve host '" + host_ + "': " + gai_strerror(gai));
        }
        addr.sin_addr = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr;
        freeaddrinfo(res);
    }

    if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("connect(): " + std::string(strerror(errno)));
    }
}

void TCPClient::disconnect() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

bool TCPClient::recv_exact(int fd, void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t r = recv(fd, p, remaining, 0);
        if (r > 0) {
            p += r;
            remaining -= static_cast<size_t>(r);
            continue;
        }
        if (r == 0) return false;       // server closed
        if (errno == EINTR) continue;
        return false;
    }
    return true;
}

bool TCPClient::send_all(int fd, const void* buf, size_t n) {
    auto* p = static_cast<const uint8_t*>(buf);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t w = send(fd, p, remaining, MSG_NOSIGNAL);
        if (w > 0) {
            p += w;
            remaining -= static_cast<size_t>(w);
            continue;
        }
        if (w < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

void TCPClient::send_request(proto::Command cmd, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> frame;
    frame.reserve(proto::HEADER_SIZE + payload.size());
    proto::write_request_header(frame, cmd, static_cast<uint32_t>(payload.size()));
    frame.insert(frame.end(), payload.begin(), payload.end());
    if (!send_all(fd_, frame.data(), frame.size())) {
        throw std::runtime_error("send failed");
    }
}

TCPClient::Response TCPClient::recv_response() {
    uint8_t header[proto::HEADER_SIZE];
    if (!recv_exact(fd_, header, proto::HEADER_SIZE)) {
        throw std::runtime_error("recv header failed");
    }

    uint16_t magic;
    std::memcpy(&magic, header, 2);
    if (magic != proto::MAGIC) {
        throw std::runtime_error("invalid response magic");
    }

    Response resp;
    resp.status = static_cast<proto::Status>(header[2]);

    uint32_t payload_len;
    std::memcpy(&payload_len, header + 3, 4);

    // Mirror the server's cap. Without this a buggy or hostile server can
    // make us allocate up to 4 GiB by sending a header with a large length.
    if (payload_len > proto::MAX_PAYLOAD_SIZE) {
        throw std::runtime_error("response payload too large");
    }

    resp.payload.resize(payload_len);
    if (payload_len > 0 && !recv_exact(fd_, resp.payload.data(), payload_len)) {
        throw std::runtime_error("recv payload failed");
    }

    return resp;
}

TCPClient::Response TCPClient::request(proto::Command cmd, const std::vector<uint8_t>& payload) {
    if (fd_ < 0) throw std::runtime_error("not connected");
    send_request(cmd, payload);
    return recv_response();
}

// ── Core operations ────────────────────────────────────────────

bool TCPClient::ping() {
    auto resp = request(proto::Command::PING);
    return resp.status == proto::Status::OK;
}

bool TCPClient::insert(const std::string& key, const float* data, size_t dims, const std::string& metadata) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_string(key);
    w.write_u32(static_cast<uint32_t>(dims));
    w.write_floats(data, dims);
    w.write_string(metadata);

    auto resp = request(proto::Command::INSERT, payload);
    return resp.status == proto::Status::OK;
}

bool TCPClient::insert(const std::string& key, const std::vector<float>& data, const std::string& metadata) {
    return insert(key, data.data(), data.size(), metadata);
}

std::vector<TCPClient::SearchResult> TCPClient::search(const float* data, size_t dims, size_t k) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_u32(static_cast<uint32_t>(dims));
    w.write_floats(data, dims);
    w.write_u32(static_cast<uint32_t>(k));

    auto resp = request(proto::Command::SEARCH, payload);
    if (resp.status != proto::Status::OK) return {};

    proto::BufferReader r(resp.payload.data(), resp.payload.size());
    uint32_t count = r.read_u32();

    std::vector<SearchResult> results;
    // Bound the reservation by the response payload (each result is >= 6 bytes),
    // so a hostile/buggy server can't drive an unbounded client allocation.
    results.reserve(std::min<size_t>(count, r.remaining() / 6 + 1));
    for (uint32_t i = 0; i < count; ++i) {
        SearchResult sr;
        sr.key = r.read_string();
        sr.distance = r.read_f32();
        results.push_back(std::move(sr));
    }
    return results;
}

std::vector<TCPClient::SearchResult> TCPClient::search(const std::vector<float>& data, size_t k) {
    return search(data.data(), data.size(), k);
}

bool TCPClient::remove(const std::string& key) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_string(key);

    auto resp = request(proto::Command::DELETE, payload);
    return resp.status == proto::Status::OK;
}

bool TCPClient::update(const std::string& key, const float* data, size_t dims, const std::string& metadata) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_string(key);
    w.write_u32(static_cast<uint32_t>(dims));
    w.write_floats(data, dims);
    w.write_string(metadata);

    auto resp = request(proto::Command::UPDATE, payload);
    return resp.status == proto::Status::OK;
}

bool TCPClient::update(const std::string& key, const std::vector<float>& data, const std::string& metadata) {
    return update(key, data.data(), data.size(), metadata);
}

std::optional<TCPClient::GetResult> TCPClient::get(const std::string& key) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_string(key);

    auto resp = request(proto::Command::GET, payload);
    if (resp.status != proto::Status::OK) return std::nullopt;

    proto::BufferReader r(resp.payload.data(), resp.payload.size());
    uint32_t dims = r.read_u32();
    if (static_cast<size_t>(dims) * sizeof(float) > r.remaining())
        throw std::runtime_error("server-declared dims exceed response payload");

    GetResult result;
    result.vector.resize(dims);
    r.read_floats(result.vector.data(), dims);
    result.metadata = r.read_string();
    return result;
}

// ── Batch operations ───────────────────────────────────────────

TCPClient::BatchResult TCPClient::batch_insert(const std::vector<std::string>& keys,
                                                const std::vector<std::vector<float>>& vectors,
                                                const std::vector<std::string>& metadata) {
    if (vectors.size() != keys.size()) {
        throw std::invalid_argument("batch_insert: keys and vectors size mismatch");
    }
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_u32(static_cast<uint32_t>(keys.size()));
    for (size_t i = 0; i < keys.size(); ++i) {
        w.write_string(keys[i]);
        w.write_u32(static_cast<uint32_t>(vectors[i].size()));
        w.write_floats(vectors[i].data(), vectors[i].size());
        w.write_string(i < metadata.size() ? metadata[i] : "");
    }

    auto resp = request(proto::Command::BATCH_INSERT, payload);
    if (resp.status != proto::Status::OK) return {0, static_cast<uint32_t>(keys.size())};

    proto::BufferReader r(resp.payload.data(), resp.payload.size());
    return {r.read_u32(), r.read_u32()};
}

TCPClient::BatchResult TCPClient::batch_delete(const std::vector<std::string>& keys) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_u32(static_cast<uint32_t>(keys.size()));
    for (const auto& key : keys) {
        w.write_string(key);
    }

    auto resp = request(proto::Command::BATCH_DELETE, payload);
    if (resp.status != proto::Status::OK) return {0, static_cast<uint32_t>(keys.size())};

    proto::BufferReader r(resp.payload.data(), resp.payload.size());
    return {r.read_u32(), r.read_u32()};
}

// ── Configuration ──────────────────────────────────────────────

bool TCPClient::set_algorithm(const std::string& algo) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_string(algo);

    auto resp = request(proto::Command::SET_ALGORITHM, payload);
    return resp.status == proto::Status::OK;
}

bool TCPClient::set_metric(const std::string& metric) {
    std::vector<uint8_t> payload;
    proto::BufferWriter w(payload);
    w.write_string(metric);

    auto resp = request(proto::Command::SET_METRIC, payload);
    return resp.status == proto::Status::OK;
}

TCPClient::Stats TCPClient::stats() {
    auto resp = request(proto::Command::STATS);
    if (resp.status != proto::Status::OK) return {};

    proto::BufferReader r(resp.payload.data(), resp.payload.size());
    Stats s;
    s.total_vectors = r.read_u32();
    s.total_searches = r.read_u32();
    s.total_inserts = r.read_u32();
    s.total_deletes = r.read_u32();
    return s;
}
