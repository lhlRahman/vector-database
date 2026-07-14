// ANN recall-vs-QPS benchmark: traces the Pareto curve for our HNSW against
// exact ground truth (FlatIndex, same metric). Works on real TEXMEX datasets
// (--data <dir> with sift_base.fvecs/sift_query.fvecs) or on synthetic clustered
// data (default), so the harness is validated even without the dataset download.
//
//   make bench-ann
//   ./build/bench_ann --data datasets/sift          # real SIFT1M
//   ./build/bench_ann --n 30000 --d 128 --q 1000    # synthetic

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../src/algorithms/flat_index.hpp"
#include "../src/algorithms/hnsw_index.hpp"
#include "../src/core/vector.hpp"
#include "../src/core/vector_accessor.hpp"
#include "../src/utils/distance_metric_policies.hpp"
#include "../src/utils/distance_metrics.hpp"
#include "../src/utils/vecs_io.hpp"

using Clock = std::chrono::steady_clock;

namespace {

struct Dataset {
    std::vector<float> base;
    std::vector<float> query;
    size_t nb = 0, nq = 0, d = 0;
};

// C gaussian clusters; base and queries drawn from them so there is genuine
// nearest-neighbor structure (uniform-random data has none and gives a
// meaningless recall signal).
Dataset make_synthetic(size_t nb, size_t nq, size_t d, size_t clusters, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> center_dist(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.15f);
    std::uniform_int_distribution<size_t> pick(0, clusters - 1);

    std::vector<float> centers(clusters * d);
    for (float& c : centers) c = center_dist(rng);

    Dataset ds;
    ds.d = d; ds.nb = nb; ds.nq = nq;
    ds.base.resize(nb * d);
    ds.query.resize(nq * d);
    auto fill = [&](std::vector<float>& out, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            size_t c = pick(rng);
            for (size_t j = 0; j < d; ++j) out[i * d + j] = centers[c * d + j] + noise(rng);
        }
    };
    fill(ds.base, nb);
    fill(ds.query, nq);
    return ds;
}

Dataset load_real(const std::string& dir) {
    namespace fs = std::filesystem;
    // Dataset-agnostic: find whatever *_base.fvecs / *_query.fvecs the dir ships
    // (sift_*, gist_*, ...), so the same loader path serves SIFT1M and GIST1M.
    std::string base_path, query_path;
    auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    for (const auto& e : fs::directory_iterator(dir)) {
        auto name = e.path().filename().string();
        if (ends_with(name, "_base.fvecs")) base_path = e.path().string();
        else if (ends_with(name, "_query.fvecs")) query_path = e.path().string();
    }
    if (base_path.empty() || query_path.empty())
        throw std::runtime_error("no *_base.fvecs / *_query.fvecs found in " + dir);
    auto base = vecs_io::load_fvecs(base_path);
    auto query = vecs_io::load_fvecs(query_path);
    if (base.d != query.d) throw std::runtime_error("base/query dimension mismatch");
    Dataset ds;
    ds.d = base.d; ds.nb = base.n; ds.nq = query.n;
    ds.base = std::move(base.data);
    ds.query = std::move(query.data);
    return ds;
}

double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>((p / 100.0) * static_cast<double>(sorted.size() - 1));
    return sorted[idx];
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir;
    size_t N = 30000, D = 128, Q = 1000, C = 150, k = 10, M = 16, efc = 200;
    unsigned seed = 42;
    std::vector<size_t> ef_list = {10, 16, 24, 32, 48, 64, 100, 128, 200, 256, 400, 500};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--data") data_dir = next();
        else if (a == "--n") N = std::stoul(next());
        else if (a == "--d") D = std::stoul(next());
        else if (a == "--q") Q = std::stoul(next());
        else if (a == "--clusters") C = std::stoul(next());
        else if (a == "--M") M = std::stoul(next());
        else if (a == "--efc") efc = std::stoul(next());
        else if (a == "--k") k = std::stoul(next());
    }

    Dataset ds;
    if (!data_dir.empty()) {
        std::cout << "Loading real dataset from " << data_dir << " ...\n";
        ds = load_real(data_dir);
    } else {
        std::cout << "Generating synthetic clustered dataset (N=" << N << " D=" << D
                  << " Q=" << Q << " clusters=" << C << ")\n";
        ds = make_synthetic(/*nb=*/N, /*nq=*/Q, /*d=*/D, /*clusters=*/C, seed);
    }
    const size_t d = ds.d;
    std::cout << "  base=" << ds.nb << "  query=" << ds.nq << "  dim=" << d << "\n";

    auto accessor = [&ds, d](uint64_t id) -> const float* {
        return ds.base.data() + static_cast<size_t>(id) * d;
    };
    std::unordered_map<std::string, uint64_t> key_to_slot;
    key_to_slot.reserve(ds.nb);
    for (size_t i = 0; i < ds.nb; ++i) key_to_slot.emplace(std::to_string(i), i);

    auto query_vec = [&](size_t q) {
        return Vector(std::vector<float>(ds.query.begin() + q * d, ds.query.begin() + (q + 1) * d));
    };

    // Ground truth. Two sources:
    //   (a) shipped exact GT (.ivecs) — mandatory for real datasets (SIFT/GIST):
    //       brute force over 1M base is ~1e12 ops, infeasible here. Only neighbor
    //       IDs are shipped (no distances) -> ID-set recall is the primary metric.
    //   (b) FlatIndex brute force — synthetic sets; gives distances too, so we can
    //       also score the tie-aware (distance-threshold) recall.
    // gt[q] holds (key, distance); distance is 0 for shipped GT (unknown).
    std::vector<std::vector<std::pair<std::string, float>>> gt(ds.nq);
    bool shipped_gt = false;
    if (!data_dir.empty()) {
        std::filesystem::path gtp;
        for (const char* n : {"sift_groundtruth.ivecs", "gist_groundtruth.ivecs", "groundtruth.ivecs"}) {
            std::filesystem::path c = std::filesystem::path(data_dir) / n;
            if (std::filesystem::exists(c)) { gtp = c; break; }
        }
        if (!gtp.empty()) {
            auto g = vecs_io::load_ivecs(gtp.string());
            std::cout << "Using shipped ground truth: " << gtp.string() << "  (" << g.n << " x " << g.d << ")\n";
            for (size_t q = 0; q < ds.nq && q < g.n; ++q) {
                const int32_t* r = g.row(q);
                for (size_t j = 0; j < k && j < g.d; ++j) gt[q].emplace_back(std::to_string(r[j]), 0.0f);
            }
            shipped_gt = true;
        }
    }
    if (!shipped_gt) {
        std::cout << "Computing exact ground truth (FlatIndex, k=" << k << ") ...\n";
        FlatIndex<EuclideanMetricPolicy> flat(d, accessor);
        auto t0 = Clock::now();
        for (size_t q = 0; q < ds.nq; ++q) gt[q] = flat.search(query_vec(q), k, key_to_slot);
        std::cout << "  GT in " << std::chrono::duration<double, std::milli>(Clock::now() - t0).count()
                  << " ms\n";
    }

    // Build the HNSW graph once; ef_search is swept at query time (no rebuild).
    std::cout << "Building HNSW (M=" << M << " ef_construction=" << efc << ") over " << ds.nb
              << " vectors ...\n";
    HNSWIndex hnsw(d, M, efc, ef_list.front(), std::make_shared<EuclideanDistance>(), accessor,
                   HNSWIndex::AllocationStrategy::Arena);  // default engine's strategy; validates the pool-resource memory fix
    double build_ms;
    {
        auto t0 = Clock::now();
        for (size_t i = 0; i < ds.nb; ++i) hnsw.insert(i, std::to_string(i));
        build_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
    double peak_mib = static_cast<double>(hnsw.getMemoryStatistics().peak_bytes_outstanding) / (1024.0 * 1024.0);
    std::cout << "  built in " << std::fixed << std::setprecision(1) << build_ms
              << " ms  (peak HNSW arena " << std::setprecision(1) << peak_mib << " MiB)\n\n";

    std::filesystem::create_directories("build/ann_results");
    std::ofstream csv("build/ann_results/results.csv");
    csv << "index,M,ef_construction,ef_search,recall_at_" << k << ",id_recall_at_" << k
        << ",qps,build_ms,peak_rss_mib,p50_us,p95_us,p99_us\n";

    std::cout << std::left << std::setw(11) << "ef_search" << std::setw(13) << ("recall@" + std::to_string(k))
              << std::setw(12) << "id_recall" << std::setw(12) << "qps" << std::setw(10) << "p50_us"
              << std::setw(10) << "p99_us" << "\n"
              << std::string(64, '-') << "\n";

    double prev_recall = -1.0, max_recall = 0.0;
    bool monotonic = true;
    for (size_t ef : ef_list) {
        if (ef < k) continue;
        hnsw.setEfSearch(ef);
        for (size_t q = 0; q < std::min<size_t>(ds.nq, 200); ++q) (void)hnsw.search(query_vec(q), k);  // warmup

        std::vector<double> lat;
        lat.reserve(ds.nq);
        size_t tie_hits = 0, id_hits = 0, total = 0;
        auto t0 = Clock::now();
        for (size_t q = 0; q < ds.nq; ++q) {
            Vector query = query_vec(q);
            auto qs = Clock::now();
            auto res = hnsw.search(query, k);
            lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - qs).count());
            const auto& g = gt[q];
            size_t kk = std::min(k, g.size());
            total += kk;
            if (kk == 0) continue;
            // Tie-aware (ann-benchmarks standard): a returned point counts if its
            // distance is within the true k-th distance (+eps) — near-tied points
            // are all valid neighbors, so ID mismatch there is not a miss. Only
            // valid when GT carries distances (FlatIndex); shipped GT has none.
            if (!shipped_gt) {
                float threshold = g[kk - 1].second * (1.0f + 1e-4f) + 1e-6f;
                for (auto& [key, dist] : res) if (dist <= threshold) ++tie_hits;
            }
            // Plain ID-set recall (the standard metric vs shipped GT).
            for (auto& [key, dist] : res)
                for (size_t gi = 0; gi < kk; ++gi) if (g[gi].first == key) { ++id_hits; break; }
        }
        double wall = std::chrono::duration<double>(Clock::now() - t0).count();
        double recall_id = total ? static_cast<double>(id_hits) / static_cast<double>(total) : 0.0;
        // Primary recall: tie-aware for FlatIndex GT, ID-set for shipped GT.
        double recall = shipped_gt ? recall_id
                                   : (total ? static_cast<double>(tie_hits) / static_cast<double>(total) : 0.0);
        double qps = wall > 0 ? static_cast<double>(ds.nq) / wall : 0.0;
        std::sort(lat.begin(), lat.end());

        max_recall = std::max(max_recall, recall);
        if (prev_recall >= 0 && recall < prev_recall - 0.02) monotonic = false;
        prev_recall = recall;

        csv << "vector-db," << M << "," << efc << "," << ef << "," << std::fixed << std::setprecision(4)
            << recall << "," << recall_id << "," << std::setprecision(1) << qps << "," << build_ms << ","
            << peak_mib << "," << percentile(lat, 50) << "," << percentile(lat, 95) << "," << percentile(lat, 99) << "\n";
        std::cout << std::left << std::setw(11) << ef << std::setw(13) << std::fixed << std::setprecision(4)
                  << recall << std::setw(12) << recall_id << std::setw(12) << std::setprecision(0) << qps
                  << std::setw(10) << std::setprecision(1) << percentile(lat, 50) << std::setw(10)
                  << percentile(lat, 99) << "\n";
    }
    csv.close();

    std::cout << "\n=== verdict (recall@" << k << ", vs "
              << (shipped_gt ? "shipped exact GT [ID recall]" : "exact FlatIndex GT [tie-aware]") << ") ===\n";
    std::cout << "max recall=" << std::fixed << std::setprecision(4) << max_recall
              << "   monotonic-with-ef=" << (monotonic ? "yes" : "NO") << "\n";
    if (!monotonic || max_recall < 0.80)
        std::cout << "  BROKEN signal: recall caps low or drops as ef rises -> inspect selectNeighbors heuristic.\n";
    else if (max_recall >= 0.97)
        std::cout << "  GOOD: recall reaches " << max_recall << " and climbs with ef -> behaves like a real HNSW.\n";
    else
        std::cout << "  WEAK: recall in [0.80,0.97) -> likely the simple neighbor-selection heuristic (consider the pruning heuristic).\n";
    std::cout << "CSV -> build/ann_results/results.csv\n";
    return 0;
}
