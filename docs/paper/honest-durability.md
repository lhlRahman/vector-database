# Honest Durability for Graph ANN — paper plan (merges theses #1 + #3 + #5)

Status: experiment plan + outline. Backbone = #1 (durability tax) with #3 (recovery)
and #5 (honest-benchmarking methodology) folded in. Upgrade path: promote #2
(recall-aware commit) to headline IF the live-web novelty check clears.

## Target venues
- **USENIX FAST** (storage/durability/recovery — best fit) · **PVLDB EA&B track** (characterization + methodology) ·
  **USENIX ATC / EuroSys** (systems). First/faster: **DaMoN** or **ADMS** workshop; **arXiv** immediately.
- Deadlines shift yearly — confirm each CFP (web blocked in this env).

## One-line thesis
ANN research measures recall and QPS but ignores durability; when you make a
graph-ANN index *actually* crash-safe, per-write fsync imposes a large, so-far-
unquantified "durability tax" — and we show how to measure it honestly, pay most
of it down with group commit, and recover fast via sealed HNSW snapshots.

## Contributions
- **C1 (methodology, from #5):** a durability-aware measurement protocol for ANN
  indexes — a tmpfs guard, an fsync-floor baseline, and mandatory fsync-mode
  labeling (plain vs `F_FULLFSYNC`). Standard suites (ANN-Benchmarks, Big ANN,
  VectorDBBench) have no crash/fsync/recovery case; this fills that gap.
- **C2 (characterization, from #1):** the first ANN-isolated **durability tax**
  curve — insert throughput / p99 / build cost vs recall as a function of the
  durability guarantee, on a real graph index (HNSW).
- **C3 (mechanism, from #1):** **group commit** for a graph-ANN WAL — one fsync
  per batch decouples commit latency from flush latency; measured speedup +
  p99 tail collapse; crash-consistent (fork+kill test).
- **C4 (recovery, from #3):** **snapshot-load vs WAL-replay** recovery analysis
  for a standalone graph ANN — recovery-time-vs-index-size and the sealed/mutable
  crossover; correctness under fault injection.

## The arc (paper structure)
1. **Intro** — ANN systems assume in-memory / rebuild-on-restart; durability is
   unmeasured. Thesis + contributions.
2. **Background** — HNSW; WAL + sealed-snapshot segmented storage (this system);
   what "durable" means (page cache vs drive-cache flush).
3. **How to measure durability honestly (C1)** — the tmpfs trap, the fsync floor,
   fsync-mode labeling; why prior ANN write numbers are optimistic.
4. **The durability tax (C2)** — per-write fsync cost across recall/QPS/p99.
5. **Group commit (C3)** — design, the recall/staleness note, results, crash test.
6. **Fast recovery (C4)** — snapshot vs replay, crossover, fault injection.
7. **Related work** — LSM-VEC (2505.17152), FreshDiskANN, SPFresh (SOSP'23),
   Starling (SIGMOD'24), DiskANN/Cosmos; Aether group commit (VLDB'10); HNSW
   (TPAMI'20); ANN-Benchmarks (Inf. Sys. 2020). Distinguish: none isolate an
   ANN fsync-tax / recovery-time-vs-size curve.
8. **Threats to validity** — dev box is macOS/Apple Silicon (plain fsync ≠ drive
   flush); numbers labeled by fsync mode; true power-loss needs Linux dm-log-writes.

## Experiment matrix
- **Datasets:** SIFT1M (128d, L2, ships GT) primary; GIST1M (960d) high-dim; a
  synthetic clustered set for controlled sweeps. (`datasets/fetch_sift.sh`.)
- **Metrics:** recall@10 (tie-aware, vs exact GT), QPS (single-thread), insert
  throughput, insert p50/p95/p99/max, index build time, peak RSS, recovery time,
  fsync floor (durable syncs/s).
- **Baselines / systems:** our engine (per-write fsync vs group commit; plain
  vs `F_FULLFSYNC`); hnswlib (recall/QPS parity check — `make bench-hnswlib`);
  LSM-VEC/FreshDiskANN/Starling as related-work positioning.
- **Media:** tmpfs (to demonstrate the trap → excluded), local NVMe (APFS),
  Linux ext4/btrfs on NVMe (honest numbers + power-loss injection).
- **Plots/tables:**
  - T1 fsync floor by media × mode (the tax's device ceiling).
  - F1 durability tax: insert QPS + p99 vs fsync mode (plain / F_FULLFSYNC / none).
  - F2 group commit: throughput vs batch size; p99 before/after (the money graph).
  - F3 recall-vs-QPS Pareto (ours vs hnswlib) — establishes the index is competitive.
  - F4 recovery time vs N: sealed (snapshot) vs mutable (WAL-replay); the crossover.

## Preliminary results already measured in this repo (M4 Pro, `-mcpu=native`)
- **Durability tax / group commit** (`make bench-durability`; synthetic, d=64):
  - fsync floor **41,101/s (plain)** vs **316/s (`F_FULLFSYNC`, ~130×)** — the honest cost.
  - insert throughput: per-write **442/s plain**, **110/s honest**; group commit **483/s**.
  - **group commit speedup 4.4×** under honest durability (110→483/s); p99 43 ms → collapsed.
- **Recovery** (`make bench-durability`, N=1500): sealed **35 ms** vs WAL-replay **225 ms**.
- **Index competitiveness** (`make bench-ann`; d=128, n=30k, M=16, efc=200):
  recall@10 **0.97 @ef=10 → 1.0 @ef=100**, monotonic; ~**97k QPS @ef=10** after
  hot-path optimizations (visited-list, FMA kernels, devirtualized distance);
  arena memory **2 GB → 38 MiB** (pool_resource).
- **Crash-consistency:** `group commit crash recovery` unit test (fork + kill,
  no clean shutdown) proves the batch's single fsync is durable; WAL now uses a
  real **CRC-32 over header+payload** (was FNV-1a, payload-only).

## Remaining work (to camera-ready)
- [ ] Real-NVMe + Linux runs (ext4/btrfs); label every number by fsync mode.
- [ ] SIFT1M/GIST1M runs (`fetch_sift.sh`) + hnswlib parity (`bench-hnswlib`).
- [ ] Concurrent group committer (leader/follower) — currently batch-API group commit.
- [ ] Fault injection at scale: torn-tail (promote fuzz_wal to a prefix assertion),
      kill -9 sweep, Linux `dm-log-writes` for true power-loss.
- [x] Real CRC-32 over header+payload (done — strengthens the crash-consistency claim).
- [x] Group commit + `F_FULLFSYNC` honest toggle + tmpfs guard + crash test (done).

## Reproducibility
`make bench-durability [DURABILITY_ARGS="--full-fsync --dir /nvme"]`,
`make bench-ann ANN_ARGS="--data datasets/sift"`, `make bench-hnswlib`,
`make test` (111 tests, ASan+UBSan+TSan clean).

## The one owed de-risk
Live-web novelty sweep (blocked in this sandbox): confirm no 2024–26 paper
publishes an ANN-isolated fsync-tax / recovery-vs-size curve, and — gating the
#2 upgrade — no recall-/freshness-quantified durability SLA for ANN.
