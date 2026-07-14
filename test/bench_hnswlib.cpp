// hnswlib baseline — recall@10 vs QPS on the SAME dataset/params as bench_ann,
// so we can check whether our HNSW is competitive. Uses the dataset's shipped
// ground truth (sift_groundtruth.ivecs) and emits the same CSV schema as
// bench_ann (index=hnswlib).
//
// Requires the vendored header (datasets/fetch_sift.sh clones it):
//   make bench-hnswlib HNSWLIB_ARGS="--data datasets/sift"
//
// NOTE: this file is compiled only via `make bench-hnswlib` (which adds
// -Ithird_party/hnswlib); it is intentionally NOT part of `make test`.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "hnswlib/hnswlib.h"      // vendored, header-only
#include "../src/utils/vecs_io.hpp"

using Clock = std::chrono::steady_clock;

static double percentile(std::vector<double>& s, double p) {
    if (s.empty()) return 0.0;
    std::sort(s.begin(), s.end());
    return s[static_cast<size_t>((p / 100.0) * static_cast<double>(s.size() - 1))];
}

int main(int argc, char** argv) {
    std::string dir;
    size_t M = 16, efc = 200, k = 10;
    std::vector<size_t> ef_list = {10, 16, 24, 32, 48, 64, 100, 128, 200, 256, 400, 500};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--data") dir = next();
        else if (a == "--M") M = std::stoul(next());
        else if (a == "--efc") efc = std::stoul(next());
        else if (a == "--k") k = std::stoul(next());
    }
    if (dir.empty()) { std::cerr << "usage: bench_hnswlib --data <dir>\n"; return 2; }
    namespace fs = std::filesystem;

    // Dataset-agnostic: find whatever *_base/_query/_groundtruth the dir ships
    // (sift_*, gist_*, ...), matching bench_ann's loader.
    std::string base_p, query_p, gt_p;
    auto ends = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    for (const auto& e : fs::directory_iterator(dir)) {
        auto n = e.path().filename().string();
        if (ends(n, "_base.fvecs")) base_p = e.path().string();
        else if (ends(n, "_query.fvecs")) query_p = e.path().string();
        else if (ends(n, "_groundtruth.ivecs")) gt_p = e.path().string();
    }
    if (base_p.empty() || query_p.empty() || gt_p.empty()) {
        std::cerr << "missing *_base/_query.fvecs or *_groundtruth.ivecs in " << dir << "\n";
        return 2;
    }
    auto base = vecs_io::load_fvecs(base_p);
    auto query = vecs_io::load_fvecs(query_p);
    auto gt = vecs_io::load_ivecs(gt_p);
    const size_t d = base.d;
    std::cout << "hnswlib baseline  base=" << base.n << " query=" << query.n << " dim=" << d
              << " gt_k=" << gt.d << "  (M=" << M << " efc=" << efc << ")\n";

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float> index(&space, base.n, M, efc);

    auto t0 = Clock::now();
    for (size_t i = 0; i < base.n; ++i) index.addPoint(base.row(i), i);
    double build_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cout << "built in " << std::fixed << std::setprecision(1) << build_ms << " ms\n\n";

    fs::create_directories("build/ann_results");
    std::ofstream csv("build/ann_results/hnswlib.csv");
    csv << "index,M,ef_construction,ef_search,recall_at_" << k << ",qps,build_ms,p50_us,p95_us,p99_us\n";
    std::cout << std::left << std::setw(11) << "ef_search" << std::setw(14) << ("recall@" + std::to_string(k))
              << std::setw(13) << "qps" << std::setw(11) << "p50_us" << std::setw(11) << "p99_us" << "\n"
              << std::string(60, '-') << "\n";

    for (size_t ef : ef_list) {
        if (ef < k) continue;
        index.setEf(ef);
        for (size_t q = 0; q < std::min<size_t>(query.n, 200); ++q) (void)index.searchKnn(query.row(q), k);  // warmup

        std::vector<double> lat;
        lat.reserve(query.n);
        size_t hits = 0, total = 0;
        auto ts = Clock::now();
        for (size_t q = 0; q < query.n; ++q) {
            auto qs = Clock::now();
            auto res = index.searchKnn(query.row(q), k);
            lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - qs).count());
            // ID-recall@k vs the shipped exact GT (top-k of the ivecs row).
            std::unordered_set<int32_t> truth;
            for (size_t j = 0; j < k && j < gt.d; ++j) truth.insert(gt.row(q)[j]);
            total += std::min(k, gt.d);
            while (!res.empty()) { if (truth.count(static_cast<int32_t>(res.top().second))) ++hits; res.pop(); }
        }
        double wall = std::chrono::duration<double>(Clock::now() - ts).count();
        double recall = total ? static_cast<double>(hits) / static_cast<double>(total) : 0.0;
        double qps = query.n / wall;
        csv << "hnswlib," << M << "," << efc << "," << ef << "," << std::fixed << std::setprecision(4)
            << recall << "," << std::setprecision(1) << qps << "," << build_ms << ","
            << percentile(lat, 50) << "," << percentile(lat, 95) << "," << percentile(lat, 99) << "\n";
        std::cout << std::left << std::setw(11) << ef << std::setw(14) << std::fixed << std::setprecision(4)
                  << recall << std::setw(13) << std::setprecision(0) << qps << std::setw(11)
                  << std::setprecision(1) << percentile(lat, 50) << std::setw(11) << percentile(lat, 99) << "\n";
    }
    std::cout << "\nCSV -> build/ann_results/hnswlib.csv (compare vs build/ann_results/results.csv AT EQUAL RECALL)\n";
    return 0;
}
