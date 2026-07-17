#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "tcp_server.hpp"

namespace {
std::mutex              g_shutdown_mtx;
std::condition_variable g_shutdown_cv;
std::atomic<bool>       g_shutdown_requested{false};
TCPServer*              g_server = nullptr;

// Signal handlers run in a restricted context. Set an atomic flag and wake
// the main thread; do not touch the server here. notify_one on a
// condition_variable is not strictly async-signal-safe, but glibc's
// implementation is in practice; the alternative (self-pipe) is heavier
// and not necessary for graceful SIGINT/SIGTERM handling.
void signal_handler(int) {
    g_shutdown_requested.store(true, std::memory_order_release);
    g_shutdown_cv.notify_all();
}
}

int main(int argc, char* argv[]) {
    size_t dims = 128;
    std::string host = "0.0.0.0";
    int port = 9090;
    size_t threads = 4;

    // Parse a numeric argument, exiting with a clear message instead of an
    // uncaught std::invalid_argument/out_of_range (which would std::terminate).
    auto parse_ul = [](const char* s, const char* name) -> unsigned long {
        try {
            size_t pos = 0;
            unsigned long v = std::stoul(s, &pos);
            if (pos != std::string(s).size()) throw std::invalid_argument("trailing characters");
            return v;
        } catch (const std::exception&) {
            std::cerr << "Invalid value for " << name << ": '" << s << "'\n";
            std::exit(2);
        }
    };

    // Simple arg parsing: --dims N --host H --port P --threads T
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dims" && i + 1 < argc)    dims = parse_ul(argv[++i], "--dims");
        else if (arg == "--host" && i + 1 < argc) host = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = static_cast<int>(parse_ul(argv[++i], "--port"));
        else if (arg == "--threads" && i + 1 < argc) threads = parse_ul(argv[++i], "--threads");
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--dims N] [--host H] [--port P] [--threads T]\n";
            return 0;
        }
    }

    if (dims == 0)                  { std::cerr << "--dims must be > 0\n"; return 2; }
    if (port < 1 || port > 65535)   { std::cerr << "--port must be in 1..65535\n"; return 2; }
    if (threads == 0)               threads = 1;

    std::cout << "Starting TCP vector database server\n"
              << "  dimensions: " << dims << "\n"
              << "  host:       " << host << "\n"
              << "  port:       " << port << "\n"
              << "  threads:    " << threads << "\n";

    TCPServer server(dims, host, port, threads);
    g_server = &server;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    server.start();

    // Block until either a signal arrives or the server stops itself.
    {
        std::unique_lock<std::mutex> lk(g_shutdown_mtx);
        g_shutdown_cv.wait(lk, [&] {
            return g_shutdown_requested.load(std::memory_order_acquire) ||
                   !server.is_running();
        });
    }

    server.stop();

    std::cout << "\nServer stopped.\n";
    return 0;
}
