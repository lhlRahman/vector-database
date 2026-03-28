#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "tcp_server.hpp"

static TCPServer* g_server = nullptr;

static void signal_handler(int) {
    if (g_server) g_server->stop();
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

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    server.start();

    // Block until stopped
    while (server.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nServer stopped.\n";
    return 0;
}
