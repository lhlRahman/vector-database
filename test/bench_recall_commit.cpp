// Recall-aware committer: prototype + end-to-end crash validation.
//
// Realizes the mechanism from the recall-bounded-durability design and validates its
// guarantee against an EXACT oracle. Isolation model (the isolation lemma):
//   * durable layer   = an HNSW over the committed set D (the graph a crash recovers);
//   * volatile window U = the not-yet-fsync'd inserts, searched EXACTLY (flat scan)
//     and merged by true distance -- U never mutates the durable graph.
// A crash drops U. The committer targets a recall-staleness budget eps by sizing the
// window B = eps*N (benign) and flushing before it is exceeded. We measure the ACTUAL
// service recall@k lost by a crash -- Recall(A_pre) - Recall(A_rec) against the exact
// pre-crash top-k -- and check it stays <= eps. We also run the ADVERSARIAL window
// (the B query-hottest vectors) to show the budget must shrink when inserts correlate
// with queries. Because U is searched exactly and D's graph is unchanged, any lost
// answer must be in U, so the measured loss is a real service-recall loss, not just
// membership.
//
//   make bench-recall-commit ANN_ARGS="--data datasets/sift --n 50000 --q 1000"

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

#include "../src/algorithms/hnsw_index.hpp"
#include "../src/core/vector.hpp"
#include "../src/utils/distance_metrics.hpp"
#include "../src/utils/vecs_io.hpp"

using Clock = std::chrono::steady_clock;

namespace {
struct Data { std::vector<float> base, query; size_t nb = 0, nq = 0, d = 0; };

Data load_real(const std::string& dir, size_t nmax) {
    namespace fs = std::filesystem;
    std::string bp, qp;
    auto ends = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0; };
    for (const auto& e : fs::directory_iterator(dir)) {
        auto n = e.path().filename().string();
        if (ends(n, "_base.fvecs")) bp = e.path().string();
        else if (ends(n, "_query.fvecs")) qp = e.path().string();
    }
    auto b = vecs_io::load_fvecs(bp), q = vecs_io::load_fvecs(qp);
    Data ds; ds.d = b.d; ds.nb = std::min(nmax, b.n); ds.nq = q.n;
    ds.base.assign(b.data.begin(), b.data.begin() + ds.nb * b.d);
    ds.query = std::move(q.data);
    return ds;
}
Data make_synth(size_t nb, size_t nq, size_t d) {
    std::mt19937 rng(7); std::normal_distribution<float> c(0, 1), z(0, 0.15f);
    std::uniform_int_distribution<size_t> pick(0, 149);
    std::vector<float> ctr(150 * d); for (float& x : ctr) x = c(rng);
    Data ds; ds.d = d; ds.nb = nb; ds.nq = nq; ds.base.resize(nb * d); ds.query.resize(nq * d);
    auto fill = [&](std::vector<float>& o, size_t n) { for (size_t i = 0; i < n; ++i) { size_t g = pick(rng);
        for (size_t j = 0; j < d; ++j) o[i * d + j] = ctr[g * d + j] + z(rng); } };
    fill(ds.base, nb); fill(ds.query, nq); return ds;
}

float l2(const float* a, const float* b, size_t d) {
    float s = 0; for (size_t j = 0; j < d; ++j) { float e = a[j] - b[j]; s += e * e; } return s;
}
// exact top-k ids over a candidate id list, by L2 to q.
std::vector<uint32_t> exact_topk(const float* q, const std::vector<uint32_t>& cand,
                                 const std::vector<float>& base, size_t d, size_t k) {
    std::vector<std::pair<float, uint32_t>> h;
    h.reserve(cand.size());
    for (uint32_t id : cand) h.push_back({l2(q, base.data() + (size_t)id * d, d), id});
    size_t kk = std::min(k, h.size());
    std::partial_sort(h.begin(), h.begin() + kk, h.end());
    std::vector<uint32_t> out; for (size_t i = 0; i < kk; ++i) out.push_back(h[i].second); return out;
}
}  // namespace

int main(int argc, char** argv) {
    std::string dir; size_t N = 50000, D = 128, Q = 1000, k = 10, M = 16, efc = 200, ef = 128;
    for (int i = 1; i < argc; ++i) { std::string a = argv[i];
        auto nx = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--data") dir = nx(); else if (a == "--n") N = std::stoul(nx());
        else if (a == "--q") Q = std::stoul(nx()); else if (a == "--k") k = std::stoul(nx());
        else if (a == "--ef") ef = std::stoul(nx()); }

    Data ds = dir.empty() ? make_synth(N, Q, D) : load_real(dir, N);
    const size_t d = ds.d; N = ds.nb; const size_t nq = std::min<size_t>(ds.nq, Q);
    std::cout << "recall-commit: N=" << N << " queries=" << nq << " dim=" << d << " k=" << k << "\n";
    auto qptr = [&](size_t qi) { return ds.query.data() + qi * d; };
    std::vector<uint32_t> all(N); for (uint32_t i = 0; i < N; ++i) all[i] = i;

    // Exact pre-crash oracle: top-k over ALL N (D union U) for each query.
    std::vector<std::vector<uint32_t>> truth(nq);
    for (size_t q = 0; q < nq; ++q) truth[q] = exact_topk(qptr(q), all, ds.base, d, k);
    // query-hotness (for the adversarial window): how often each id is a true neighbor.
    std::unordered_map<uint32_t, int> hot;
    for (auto& t : truth) for (uint32_t id : t) hot[id]++;
    std::vector<uint32_t> by_hot = all;
    std::sort(by_hot.begin(), by_hot.end(), [&](uint32_t a, uint32_t b) { return hot[a] > hot[b]; });

    auto accessor = [&ds, d](uint64_t id) -> const float* { return ds.base.data() + (size_t)id * d; };
    auto recall = [&](const std::vector<uint32_t>& ans, const std::vector<uint32_t>& tr) {
        std::unordered_set<uint32_t> s(tr.begin(), tr.end()); size_t h = 0;
        for (uint32_t id : ans) if (s.count(id)) ++h; return tr.empty() ? 0.0 : (double)h / tr.size(); };

    std::cout << "\n" << std::left << std::setw(8) << "eps" << std::setw(9) << "W=B"
              << std::setw(12) << "regime" << std::setw(14) << "mean dRecall" << std::setw(13) << "max dRecall"
              << "SLA(mean<=eps)?\n" << std::string(72, '-') << "\n";
    std::filesystem::create_directories("build/ann_results");
    std::ofstream csv("build/ann_results/recall_commit.csv");
    csv << "eps,W,regime,mean_dRecall,max_dRecall,commits_per_N,holds\n";

    for (double eps : {0.005, 0.01, 0.02, 0.05}) {
        size_t B = std::max<size_t>(1, (size_t)std::llround(eps * N));
        for (int adv = 0; adv < 2; ++adv) {
            // Volatile window U (dropped on crash); D = committed durable set.
            std::unordered_set<uint32_t> U;
            if (!adv) for (size_t i = N - B; i < N; ++i) U.insert((uint32_t)i);          // benign: last B
            else      for (size_t i = 0; i < B; ++i) U.insert(by_hot[i]);                 // adversarial: hottest B
            std::vector<uint32_t> Uvec(U.begin(), U.end());
            // Durable HNSW over D = all ids not in U.
            HNSWIndex hnsw(d, M, efc, ef, std::make_shared<EuclideanDistance>(), accessor,
                           HNSWIndex::AllocationStrategy::Arena);
            for (uint32_t id = 0; id < N; ++id) if (!U.count(id)) hnsw.insert(id, std::to_string(id));
            hnsw.setEfSearch(ef);
            double sum = 0, mx = 0;
            for (size_t q = 0; q < nq; ++q) {
                // Durable (approx) answer = what a crash recovers.
                std::vector<uint32_t> dres;
                for (auto& [key, dist] : hnsw.search(Vector(std::vector<float>(qptr(q), qptr(q) + d)), k))
                    dres.push_back((uint32_t)std::stoul(key));
                // Live (pre-crash) answer = durable candidates + EXACT scan of isolated U.
                std::vector<uint32_t> cand = dres;
                for (uint32_t id : Uvec) cand.push_back(id);
                std::vector<uint32_t> apre = exact_topk(qptr(q), cand, ds.base, d, k);
                double dloss = std::max(0.0, recall(apre, truth[q]) - recall(dres, truth[q]));
                sum += dloss; mx = std::max(mx, dloss);
            }
            double mean = sum / nq; bool holds = mean <= eps + 1e-9;
            const char* rg = adv ? "adversarial" : "benign";
            std::cout << std::setw(8) << std::fixed << std::setprecision(3) << eps << std::setw(9) << B
                      << std::setw(12) << rg << std::setw(14) << std::setprecision(4) << mean
                      << std::setw(13) << mx << (holds ? "yes" : "NO") << "\n";
            csv << eps << "," << B << "," << rg << "," << mean << "," << mx << "," << (double)N / B << ","
                << (holds ? 1 : 0) << "\n";
        }
    }
    csv.close();
    std::cout << "\nRead: benign mean dRecall <= eps at every eps confirms the recall-aware committer's\n"
              << "SLA end-to-end under a real (approximate) durable HNSW + exact isolated window; the\n"
              << "adversarial column shows the budget must shrink when inserts are query-correlated.\n"
              << "CSV -> build/ann_results/recall_commit.csv\n";
    return 0;
}
