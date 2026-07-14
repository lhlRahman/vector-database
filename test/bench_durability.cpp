// Durability benchmark — the "durability tax on ANN" measurement.
//
//   make bench-durability                                  # plain fsync, d=128
//   make bench-durability DURABILITY_ARGS=--full-fsync     # honest macOS durability
//   make bench-durability DURABILITY_ARGS="--dir /path/on/nvme --trials 7 --d 128"
//
// Reports, on a REAL (non-tmpfs) filesystem and as MEDIAN (min-max) over K trials:
//   * the fsync floor (durable syncs/s),
//   * per-write-fsync insert throughput/latency (the durability tax),
//   * a group-commit BATCH-SIZE SWEEP (throughput/speedup vs. batch size), so the
//     speedup is reported as a curve, not a single operating point, and
//   * cold-open recovery time for the sealed (snapshot) vs mutable (WAL-replay)
//     paths across a sweep of N.
// Refuses to report write numbers on tmpfs, where fsync is a no-op (the trap our
// own docs flag).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
#include "../src/utils/vecs_io.hpp"

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

double pct(std::vector<double> s, double p) {
    if (s.empty()) return 0.0;
    std::sort(s.begin(), s.end());
    return s[static_cast<size_t>((p / 100.0) * static_cast<double>(s.size() - 1))];
}

// Median and range across trials — the honest summary for a small K.
struct Stat { double med = 0, lo = 0, hi = 0; };
Stat summarize(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    s.med = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.lo = v.front();
    s.hi = v.back();
    return s;
}
std::string fmt(const Stat& s, int prec = 0) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << s.med << " (" << s.lo << "-" << s.hi << ")";
    return o.str();
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
    std::string dir_arg, data_arg, recn_arg, tag_arg;
    bool full = false;
    size_t N = 2000, D = 128, TRIALS = 7;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--full-fsync") full = true;
        else if (a == "--dir" && i + 1 < argc) dir_arg = argv[++i];
        else if (a == "--data" && i + 1 < argc) data_arg = argv[++i];
        else if (a == "--recn" && i + 1 < argc) recn_arg = argv[++i];   // e.g. "1000,10000,50000"
        else if (a == "--tag" && i + 1 < argc) tag_arg = argv[++i];     // CSV filename suffix
        else if (a == "--n" && i + 1 < argc) N = std::stoul(argv[++i]);
        else if (a == "--d" && i + 1 < argc) D = std::stoul(argv[++i]);
        else if (a == "--trials" && i + 1 < argc) TRIALS = std::stoul(argv[++i]);
    }
    if (full) vdb::io::set_full_fsync(true);

    // Optional: draw the insert stream from a REAL embedding set (*_base.fvecs)
    // instead of synthetic uniform-random vectors, so the tax is measured on real
    // data. D is then set by the file's dimensionality.
    std::vector<float> real_data;
    size_t real_n = 0;
    if (!data_arg.empty()) {
        std::string base_p;
        for (const auto& e : std::filesystem::directory_iterator(data_arg)) {
            auto n = e.path().filename().string();
            if (n.size() >= 11 && n.compare(n.size() - 11, 11, "_base.fvecs") == 0) { base_p = e.path().string(); break; }
        }
        if (base_p.empty()) { std::cerr << "no *_base.fvecs in " << data_arg << "\n"; return 2; }
        auto m = vecs_io::load_fvecs(base_p);            // may be large; we only need a slice
        real_n = std::min<size_t>(m.n, 20000);            // enough to cover N and recovery sizes
        D = m.d;
        real_data.assign(m.data.begin(), m.data.begin() + real_n * D);
        std::cout << "using real embeddings from " << base_p << " (" << real_n << " x " << D << ")\n";
    }
    auto sample = [&](size_t i, std::mt19937& rng) -> std::vector<float> {
        if (real_n) { const float* p = real_data.data() + (i % real_n) * D; return std::vector<float>(p, p + D); }
        return rand_vec(D, rng);
    };

    std::filesystem::path root = dir_arg.empty()
        ? (std::filesystem::temp_directory_path() / ("durability_bench_" + std::to_string(::getpid())))
        : std::filesystem::path(dir_arg) / ("durability_bench_" + std::to_string(::getpid()));
    std::filesystem::create_directories(root);

    std::cout << "Durability benchmark\n  dir: " << root << "\n  fsync: "
              << (full ? "F_FULLFSYNC (honest, macOS drive-cache flush)" : "plain fsync")
              << "\n  N=" << N << " D=" << D << " trials=" << TRIALS << "\n";
    if (is_tmpfs(root)) {
        std::cout << "\n!! " << root << " is tmpfs (RAM-backed) — fsync is a no-op here.\n"
                  << "!! Write/durability numbers would be FICTION. Re-run with --dir on a real disk.\n";
        std::filesystem::remove_all(root);
        return 2;
    }

    std::mt19937 rng(1234);
    auto make_db = [&](const std::string& sub) {
        auto p = root / sub;
        std::filesystem::remove_all(p);
        auto db = std::make_unique<VectorDatabase>(
            D, VectorDatabase::SearchMode::HNSW, false, /*batch=*/true,
            PersistenceConfig{}, false, 0, p.string(), VectorDatabase::StorageEngine::Segmented);
        db->initialize();
        return db;
    };

    const std::vector<size_t> batch_sizes = {1, 10, 50, 200, 1000, N};
    std::vector<size_t> rec_N = {1000, 3000, 6000};
    if (!recn_arg.empty()) {                      // override recovery-N sweep, e.g. "1000,10000,50000"
        rec_N.clear();
        size_t pos = 0, comma;
        do { comma = recn_arg.find(',', pos);
             rec_N.push_back(std::stoul(recn_arg.substr(pos, comma - pos)));
             pos = comma + 1; } while (comma != std::string::npos);
    }

    std::vector<double> floor_s, pw_qps, pw_p50, pw_p99, pw_max;
    std::map<size_t, std::vector<double>> gc_qps;                 // batch size -> qps/trial
    std::map<size_t, std::vector<double>> sealed_ms, mutable_ms;  // N -> ms/trial

    for (size_t t = 0; t < TRIALS; ++t) {
        std::cout << "  trial " << (t + 1) << "/" << TRIALS << " ..." << std::flush;

        floor_s.push_back(fsync_floor(root, static_cast<int>(std::min<size_t>(N, 500)), full));

        // 1) Per-write fsync: N single inserts, each fsync'd before returning.
        {
            auto db = make_db("perwrite");
            std::vector<double> lat;
            lat.reserve(N);
            auto t0 = Clock::now();
            for (size_t i = 0; i < N; ++i) {
                auto v = sample(i, rng);
                auto s = Clock::now();
                (void)db->insert(Vector(v), "k" + std::to_string(i));
                lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - s).count());
            }
            double sec = std::chrono::duration<double>(Clock::now() - t0).count();
            pw_qps.push_back(N / sec);
            pw_p50.push_back(pct(lat, 50));
            pw_p99.push_back(pct(lat, 99));
            pw_max.push_back(pct(lat, 100));
            db->shutdown();
        }

        // 2) Group-commit batch-size sweep: insert N via batches of size b, one
        // fsync per batch. b=1 approximates per-write; b=N is a single big batch.
        for (size_t b : batch_sizes) {
            auto db = make_db("gc");
            auto t0 = Clock::now();
            size_t committed = 0;
            for (size_t start = 0; start < N; start += b) {
                size_t end = std::min(N, start + b);
                std::vector<std::string> keys;
                std::vector<Vector> vecs;
                keys.reserve(end - start);
                vecs.reserve(end - start);
                for (size_t i = start; i < end; ++i) {
                    keys.push_back("k" + std::to_string(i));
                    vecs.emplace_back(sample(i, rng));
                }
                auto r = db->batchInsert(keys, vecs);
                committed += r.operations_committed;
            }
            double sec = std::chrono::duration<double>(Clock::now() - t0).count();
            gc_qps[b].push_back(sec > 0 ? committed / sec : 0.0);
            db->shutdown();
        }

        // 3) Recovery: sealed (snapshot load) vs mutable (WAL replay), swept over N.
        for (size_t n : rec_N) {
            for (const char* mode : {"sealed", "mutable"}) {
                auto p = root / (std::string("recover_") + mode + "_" + std::to_string(n));
                std::filesystem::remove_all(p);
                {
                    auto db = std::make_unique<VectorDatabase>(
                        D, VectorDatabase::SearchMode::HNSW, false, true, PersistenceConfig{}, false, 0,
                        p.string(), VectorDatabase::StorageEngine::Segmented);
                    db->initialize();
                    std::vector<std::string> keys;
                    std::vector<Vector> vecs;
                    for (size_t i = 0; i < n; ++i) { keys.push_back("k" + std::to_string(i)); vecs.emplace_back(sample(i, rng)); }
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
                double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                (std::string(mode) == "sealed" ? sealed_ms : mutable_ms)[n].push_back(ms);
                std::filesystem::remove_all(p);
            }
        }
        std::cout << " done\n";
    }

    // ---- Report: median (min-max) over trials ----------------------------------
    auto floor = summarize(floor_s);
    std::cout << "\n=== fsync floor ===\n"
              << "  " << fmt(floor, 0) << " durable syncs/s  [" << (full ? "F_FULLFSYNC" : "plain fsync") << "]\n";

    auto qps = summarize(pw_qps), p50 = summarize(pw_p50), p99 = summarize(pw_p99), mx = summarize(pw_max);
    std::cout << "\n=== durability tax (per-write fsync) ===\n"
              << "  throughput : " << fmt(qps, 0) << " ins/s\n"
              << "  p50 latency: " << fmt(p50, 0) << " us\n"
              << "  p99 latency: " << fmt(p99, 0) << " us\n"
              << "  max latency: " << fmt(mx, 0) << " us\n";

    auto gc1 = summarize(gc_qps[1]);
    std::cout << "\n=== group commit batch-size sweep (one fsync per batch) ===\n"
              << "  " << std::setw(8) << "batch" << std::setw(22) << "ins/s (med,min-max)" << std::setw(12) << "speedup\n";
    for (size_t b : batch_sizes) {
        auto s = summarize(gc_qps[b]);
        double speedup = gc1.med > 0 ? s.med / gc1.med : 0.0;
        std::cout << "  " << std::setw(8) << b << std::setw(22) << fmt(s, 0)
                  << std::setw(10) << std::fixed << std::setprecision(1) << speedup << "x\n";
    }

    std::cout << "\n=== recovery: sealed (snapshot) vs mutable (WAL replay) ===\n"
              << "  " << std::setw(8) << "N" << std::setw(22) << "sealed ms" << std::setw(22) << "mutable ms" << std::setw(8) << "gap\n";
    for (size_t n : rec_N) {
        auto se = summarize(sealed_ms[n]), mu = summarize(mutable_ms[n]);
        double gap = se.med > 0 ? mu.med / se.med : 0.0;
        std::cout << "  " << std::setw(8) << n << std::setw(22) << fmt(se, 1) << std::setw(22) << fmt(mu, 1)
                  << std::setw(6) << std::fixed << std::setprecision(1) << gap << "x\n";
    }

    // Machine-readable summary (one block; easy to transcribe into paper tables).
    std::filesystem::create_directories("build/durability_results");
    std::string tag = std::string(full ? "full" : "plain") + (real_n ? "_real" : "") +
                      (tag_arg.empty() ? "" : "_" + tag_arg);
    std::ofstream csv("build/durability_results/durability_" + tag + "_d" + std::to_string(D) + ".csv");
    csv << "metric,d,mode,median,min,max\n";
    auto row = [&](const char* m, const Stat& s) {
        csv << m << "," << D << "," << tag << "," << s.med << "," << s.lo << "," << s.hi << "\n";
    };
    row("fsync_floor", floor);
    row("perwrite_qps", qps);
    row("perwrite_p50", p50);
    row("perwrite_p99", p99);
    row("perwrite_max", mx);
    for (size_t b : batch_sizes) { std::string m = "gc_qps_b" + std::to_string(b); row(m.c_str(), summarize(gc_qps[b])); }
    for (size_t n : rec_N) { std::string m = "sealed_ms_n" + std::to_string(n); row(m.c_str(), summarize(sealed_ms[n])); }
    for (size_t n : rec_N) { std::string m = "mutable_ms_n" + std::to_string(n); row(m.c_str(), summarize(mutable_ms[n])); }
    csv.close();

    std::filesystem::remove_all(root);
    std::cout << "\nCSV -> build/durability_results/durability_" << tag << "_d" << D << ".csv\ndone.\n";
    return 0;
}
