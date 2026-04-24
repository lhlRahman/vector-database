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

    // Simple arg parsing: --dims N --host H --port P --threads T
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dims" && i + 1 < argc)    dims = std::stoul(argv[++i]);
        else if (arg == "--host" && i + 1 < argc) host = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoul(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--dims N] [--host H] [--port P] [--threads T]\n";
            return 0;
        }
    }

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
