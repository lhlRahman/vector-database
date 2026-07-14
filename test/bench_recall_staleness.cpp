// Recall-bounded durability (the ANN-native contribution).
//
// Group commit leaves a window of W acknowledged-but-not-yet-fsync'd inserts. If
// the machine crashes, those W most-recent vectors vanish. This harness measures
// the "recall at risk": the fraction of a query's top-k results that live in that
// un-durable window (i.e. would be lost by a crash), swept over W. We compare:
//   * empirical recall-at-risk on a benign query set (queries uncorrelated with
//     insertion order) -- the realistic case, ~W/N;
//   * the distribution-free worst-case bound min(W,k)/k -- what you can *prove*
//     with no workload assumption;
// and emit a CSV that, crossed with the group-commit throughput sweep, gives the
// throughput-vs-recall-staleness Pareto.
//
//   make bench-recall-staleness ANN_ARGS="--data datasets/sift"
//
// No ground truth needed: recall-at-risk is about which of the LIVE index's
// results are non-durable, not about correctness vs. an oracle. We query at high
// ef so the returned neighbors are near-exact.

#include <algorithm>
#include <chrono>
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

#include "../src/algorithms/hnsw_index.hpp"
#include "../src/core/vector.hpp"
#include "../src/utils/distance_metrics.hpp"
#include "../src/utils/vecs_io.hpp"

using Clock = std::chrono::steady_clock;

namespace {

struct Dataset {
    std::vector<float> base, query;
    size_t nb = 0, nq = 0, d = 0;
};

Dataset make_synthetic(size_t nb, size_t nq, size_t d, size_t clusters, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> center(0.f, 1.f), noise(0.f, 0.15f);
    std::uniform_int_distribution<size_t> pick(0, clusters - 1);
    std::vector<float> centers(clusters * d);
    for (float& c : centers) c = center(rng);
    Dataset ds; ds.d = d; ds.nb = nb; ds.nq = nq;
    ds.base.resize(nb * d); ds.query.resize(nq * d);
    auto fill = [&](std::vector<float>& out, size_t n) {
        for (size_t i = 0; i < n; ++i) { size_t c = pick(rng);
            for (size_t j = 0; j < d; ++j) out[i * d + j] = centers[c * d + j] + noise(rng); }
    };
    fill(ds.base, nb); fill(ds.query, nq);
    return ds;
}

Dataset load_real(const std::string& dir) {
    namespace fs = std::filesystem;
    std::string base_p, query_p;
    auto ends = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0; };
    for (const auto& e : fs::directory_iterator(dir)) {
        auto n = e.path().filename().string();
        if (ends(n, "_base.fvecs")) base_p = e.path().string();
        else if (ends(n, "_query.fvecs")) query_p = e.path().string();
    }
    if (base_p.empty() || query_p.empty()) throw std::runtime_error("no *_base/_query.fvecs in " + dir);
    auto b = vecs_io::load_fvecs(base_p); auto q = vecs_io::load_fvecs(query_p);
    Dataset ds; ds.d = b.d; ds.nb = b.n; ds.nq = q.n;
    ds.base = std::move(b.data); ds.query = std::move(q.data);
    return ds;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_dir;
    size_t N = 200000, D = 128, Q = 1000, k = 10, M = 16, efc = 200, ef = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--data") data_dir = nx();
        else if (a == "--n") N = std::stoul(nx());
        else if (a == "--q") Q = std::stoul(nx());
        else if (a == "--k") k = std::stoul(nx());
        else if (a == "--ef") ef = std::stoul(nx());
    }

    Dataset ds = data_dir.empty() ? make_synthetic(N, Q, D, 150, 42) : load_real(data_dir);
    const size_t d = ds.d;
    const size_t nq = std::min<size_t>(ds.nq, 1000);
    std::cout << "recall-staleness: base=" << ds.nb << " query=" << nq << " dim=" << d
              << " k=" << k << " ef=" << ef << "\n";

    auto accessor = [&ds, d](uint64_t id) -> const float* { return ds.base.data() + static_cast<size_t>(id) * d; };
    HNSWIndex hnsw(d, M, efc, ef, std::make_shared<EuclideanDistance>(), accessor, HNSWIndex::AllocationStrategy::Arena);
    std::cout << "building HNSW over " << ds.nb << " vectors (insertion order = id order) ...\n";
    auto t0 = Clock::now();
    for (size_t i = 0; i < ds.nb; ++i) hnsw.insert(i, std::to_string(i));
    std::cout << "  built in " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(Clock::now() - t0).count() << " ms\n";
    hnsw.setEfSearch(ef);

    // For each query, record the ids of its top-k (near-exact at this ef).
    std::vector<std::vector<uint64_t>> res_ids(nq);
    for (size_t q = 0; q < nq; ++q) {
        Vector qv(std::vector<float>(ds.query.begin() + q * d, ds.query.begin() + (q + 1) * d));
        for (auto& [key, dist] : hnsw.search(qv, k)) res_ids[q].push_back(std::stoull(key));
    }

    // Adversarial model: recent inserts ARE the query-relevant vectors. Rank ids by
    // "hotness" (how many query top-k lists they appear in); the size-W adversarial
    // window is the W hottest ids. This is the realistic worst case ("query the fresh
    // data"), bridging the benign W/N and the distribution-free min(W,k)/k.
    std::unordered_map<uint64_t, int> freq;
    for (size_t q = 0; q < nq; ++q) for (uint64_t id : res_ids[q]) freq[id]++;
    std::vector<std::pair<uint64_t, int>> hot(freq.begin(), freq.end());
    std::sort(hot.begin(), hot.end(), [](auto& a, auto& b) { return a.second > b.second; });

    // Sweep W (size of the un-durable "recent inserts" window = last W ids).
    std::vector<size_t> Wsweep = {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000};
    std::filesystem::create_directories("build/ann_results");
    std::ofstream csv("build/ann_results/recall_staleness.csv");
    csv << "W,W_frac_of_N,recall_at_risk_empirical,recall_at_risk_adversarial,worst_case_bound_min_Wk_over_k\n";
    std::cout << "\n" << std::left << std::setw(9) << "W" << std::setw(14) << "benign"
              << std::setw(14) << "adversarial" << std::setw(12) << "worst-case" << "W/N\n"
              << std::string(56, '-') << "\n";
    std::unordered_set<uint64_t> hotset; size_t hot_added = 0;
    for (size_t W : Wsweep) {
        if (W > ds.nb) break;
        uint64_t threshold = ds.nb - W;  // ids >= threshold are in the un-durable window (benign, by insert order)
        while (hot_added < W && hot_added < hot.size()) hotset.insert(hot[hot_added++].first);  // adversarial window
        double hits = 0, adv_hits = 0; size_t total = 0;
        for (size_t q = 0; q < nq; ++q) {
            for (uint64_t id : res_ids[q]) {
                if (id >= threshold) hits += 1.0;
                if (hotset.count(id)) adv_hits += 1.0;
            }
            total += res_ids[q].size();
        }
        double rar = total ? hits / static_cast<double>(total) : 0.0;
        double adv = total ? adv_hits / static_cast<double>(total) : 0.0;
        double wc = std::min<double>(W, k) / static_cast<double>(k);
        double frac = static_cast<double>(W) / static_cast<double>(ds.nb);
        csv << W << "," << frac << "," << rar << "," << adv << "," << wc << "\n";
        std::cout << std::setw(9) << W << std::scientific << std::setprecision(2)
                  << std::setw(14) << rar << std::setw(14) << adv
                  << std::fixed << std::setprecision(3) << std::setw(12) << wc
                  << std::scientific << std::setprecision(2) << frac << "\n";
    }
    csv.close();
    std::cout << "\nCSV -> build/ann_results/recall_staleness.csv\n"
              << "Read: at batch size B, a crash risks ~recall@risk(B) of each query's recall@" << k
              << ", far below the provable worst case for a benign (insert-order-uncorrelated) workload.\n";
    return 0;
}
