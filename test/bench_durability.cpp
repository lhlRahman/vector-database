// Durability benchmark — the "durability tax on ANN" measurement.
//
//   make bench-durability                         # plain fsync
//   make bench-durability DURABILITY_ARGS=--full-fsync   # honest macOS durability
//   make bench-durability DURABILITY_ARGS="--dir /path/on/nvme"
//
// Reports, on a REAL (non-tmpfs) filesystem: the fsync floor, per-write-fsync
// insert throughput/latency, group-commit (batched fsync) throughput, and cold-
// open recovery time for the sealed (snapshot) vs mutable (WAL-replay) paths.
// Refuses to report write numbers on tmpfs, where fsync is a no-op (the trap our
// own docs flag).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/mount.h>
#include <sys/param.h>
#elif defined(__linux__)
#include <sys/vfs.h>
#endif

#include "../src/core/vector_database.hpp"
#include "../src/utils/atomic_write.hpp"

using Clock = std::chrono::steady_clock;

namespace {

bool is_tmpfs(const std::filesystem::path& p) {
#if defined(__APPLE__)
    struct statfs s;
    if (statfs(p.c_str(), &s) == 0) return std::string(s.f_fstypename) == "tmpfs";
#elif defined(__linux__)
    struct statfs s;
    if (statfs(p.c_str(), &s) == 0) return s.f_type == 0x01021994;  // TMPFS_MAGIC
#endif
    (void)p;
    return false;
}

std::vector<float> rand_vec(size_t d, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(d);
    for (float& x : v) x = dist(rng);
    return v;
}

double pct(std::vector<double>& s, double p) {
    if (s.empty()) return 0.0;
    std::sort(s.begin(), s.end());
    return s[static_cast<size_t>((p / 100.0) * static_cast<double>(s.size() - 1))];
}

// Tight write()+fsync loop: how many durable syncs/sec the device sustains.
double fsync_floor(const std::filesystem::path& dir, int n, bool full) {
    auto f = dir / "fsync_floor.bin";
    int fd = ::open(f.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 0.0;
    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) {
        (void)!::write(fd, &i, sizeof(i));
#if defined(__APPLE__)
        if (full) { if (::fcntl(fd, F_FULLFSYNC) != 0) ::fsync(fd); }
        else ::fsync(fd);
#else
        ::fsync(fd);
        (void)full;
#endif
    }
    double sec = std::chrono::duration<double>(Clock::now() - t0).count();
    ::close(fd);
    std::filesystem::remove(f);
    return sec > 0 ? n / sec : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir_arg;
    bool full = false;
    size_t N = 2000, D = 64;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--full-fsync") full = true;
        else if (a == "--dir" && i + 1 < argc) dir_arg = argv[++i];
        else if (a == "--n" && i + 1 < argc) N = std::stoul(argv[++i]);
    }
    if (full) vdb::io::set_full_fsync(true);

    std::filesystem::path root = dir_arg.empty()
        ? (std::filesystem::temp_directory_path() / ("durability_bench_" + std::to_string(::getpid())))
        : std::filesystem::path(dir_arg) / ("durability_bench_" + std::to_string(::getpid()));
    std::filesystem::create_directories(root);

    std::cout << "Durability benchmark\n  dir: " << root << "\n  fsync: "
              << (full ? "F_FULLFSYNC (honest, macOS drive-cache flush)" : "plain fsync")
              << "\n  N=" << N << " D=" << D << "\n";
    if (is_tmpfs(root)) {
        std::cout << "\n!! " << root << " is tmpfs (RAM-backed) — fsync is a no-op here.\n"
                  << "!! Write/durability numbers would be FICTION. Re-run with --dir on a real disk.\n";
        std::filesystem::remove_all(root);
        return 2;
    }

    std::mt19937 rng(1234);
    auto make_db = [&](const std::string& sub) {
        auto p = root / sub;
        auto db = std::make_unique<VectorDatabase>(
            D, VectorDatabase::SearchMode::HNSW, false, /*batch=*/true,
            PersistenceConfig{}, false, 0, p.string(), VectorDatabase::StorageEngine::Segmented);
        db->initialize();
        return db;
    };

    std::cout << "\nfsync floor: " << std::fixed << std::setprecision(0)
              << fsync_floor(root, static_cast<int>(std::min<size_t>(N, 500)), full) << " durable syncs/s\n";

    // 1) Per-write fsync: N single inserts, each fsync'd before returning.
    double perwrite_qps = 0;
    {
        auto db = make_db("perwrite");
        std::vector<double> lat;
        lat.reserve(N);
        auto t0 = Clock::now();
        for (size_t i = 0; i < N; ++i) {
            auto v = rand_vec(D, rng);
            auto s = Clock::now();
            (void)db->insert(Vector(v), "k" + std::to_string(i));
            lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - s).count());
        }
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        perwrite_qps = N / sec;
        std::cout << "\n[per-write fsync]  " << std::setprecision(0) << perwrite_qps << " ins/s"
                  << "   p50=" << std::setprecision(1) << pct(lat, 50) << "us"
                  << " p99=" << pct(lat, 99) << "us max=" << pct(lat, 100) << "us\n";
        db->shutdown();
    }

    // 2) Group commit: one batchInsert of N -> a single fsync for the batch.
    double batch_qps = 0;
    {
        auto db = make_db("groupcommit");
        std::vector<std::string> keys;
        std::vector<Vector> vecs;
        keys.reserve(N); vecs.reserve(N);
        for (size_t i = 0; i < N; ++i) { keys.push_back("k" + std::to_string(i)); vecs.emplace_back(rand_vec(D, rng)); }
        auto t0 = Clock::now();
        auto r = db->batchInsert(keys, vecs);
        double sec = std::chrono::duration<double>(Clock::now() - t0).count();
        batch_qps = r.operations_committed / sec;
        std::cout << "[group commit  ]  " << std::setprecision(0) << batch_qps << " ins/s"
                  << "   (" << r.operations_committed << " in " << std::setprecision(3) << sec << "s)\n";
        db->shutdown();
    }
    std::cout << "  -> group commit speedup: " << std::setprecision(1)
              << (perwrite_qps > 0 ? batch_qps / perwrite_qps : 0.0) << "x\n";

    // 3) Recovery time: sealed (snapshot load) vs mutable (WAL replay).
    for (const char* mode : {"sealed", "mutable"}) {
        auto p = root / (std::string("recover_") + mode);
        {
            auto db = std::make_unique<VectorDatabase>(
                D, VectorDatabase::SearchMode::HNSW, false, true, PersistenceConfig{}, false, 0,
                p.string(), VectorDatabase::StorageEngine::Segmented);
            db->initialize();
            std::vector<std::string> keys; std::vector<Vector> vecs;
            for (size_t i = 0; i < N; ++i) { keys.push_back("k" + std::to_string(i)); vecs.emplace_back(rand_vec(D, rng)); }
            (void)db->batchInsert(keys, vecs);
            if (std::string(mode) == "sealed") db->sealMutableSegment();
            (void)db->checkpoint();
            db->shutdown();
        }
        auto t0 = Clock::now();
        {
            VectorDatabase db(D, VectorDatabase::SearchMode::HNSW, false, true, PersistenceConfig{}, false, 0,
                              p.string(), VectorDatabase::StorageEngine::Segmented);
            db.initialize();
            volatile size_t c = db.vectorCount();
            (void)c;
            db.shutdown();
        }
        std::cout << "[recovery " << std::setw(7) << std::left << mode << "] "
                  << std::setprecision(1) << std::chrono::duration<double, std::milli>(Clock::now() - t0).count()
                  << " ms (cold open of " << N << " vectors)\n";
    }

    std::filesystem::remove_all(root);
    std::cout << "\ndone.\n";
    return 0;
}
