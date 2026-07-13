# Audit fix tracker

Branch `fix/audit-findings`. Findings from the max-effort code review (58 verified)
+ toolchain/build fixes. Status per item; `make test` kept green after every batch.

Baseline: 100/100 (58 unit + 22 e2e + 20 tcp). Current: see latest batch.

## Done
- **#0 build** — `e2e_tests.cpp` missing `<unistd.h>` for `getpid()` (macOS build break). FIXED.
- **#6/#14 durability** — `atomic_write.hpp` `fsync_file`/`fsync_dir` swallowed open/fsync failures → reported durable on error. Now throw on failure (EINTR-retry; tolerate dir-fsync EINVAL/ENOTSUP). FIXED.
- **#5 concurrency** — `atomic_write` used a fixed `<path>.tmp`; concurrent writers collided. Now unique per-process/per-call temp name. FIXED.
- **#33 correctness** — SIMD x86 path emitted AVX with no `__AVX__` guard (SIGILL risk). Now gated on `__AVX__`, else scalar. FIXED.
- **#56 correctness** — `simd_operations.cpp` used `std::abs(float)` without `<cmath>`. Added include. FIXED.
- **#3 correctness (HIGH)** — QueryCache keyed on vector only; different `k` returned stale count. Now keyed k-aware (hit iff cached k' >= k, return prefix). FIXED + regression test.
- **#19 correctness** — `QueryCache(0)` → `back()` on empty list (UB). Now zero-capacity is a no-op. FIXED + regression test.
- **#27 correctness** — `executeBatchInsert/Update` indexed `vectors[i]` without size check. Now rejects size mismatch. FIXED.
- **#2 correctness (HIGH)** — `update()`/`batchUpdate()` accepted NaN vectors (insert rejected them). Added `containsNaN` guard to update + all batch paths. FIXED + regression test.
- **#16 correctness** — `gpuAcceleratedSearch` always Euclidean; now falls back to metric-correct CPU path for non-Euclidean. FIXED.
- **checkpoint() concurrency** — took no lock while `flush()` did; now takes a ReadGuard. FIXED.
- **segmented batch validation** — segmented `batchInsert`/`batchUpdate` skipped the dim/NaN checks the single-insert path does; added. FIXED.
- **#17 durability (HIGH)** — `setMetric()` migrated via `insertRecovered()` (no WAL) → migrated data lost on reload. Now uses `insert()` (WAL+fsync). FIXED + regression test (survives restart).

Test count: 104 (60 unit + 24 e2e + 20 tcp), all green.

## All correctness/security/durability findings FIXED
Later batches (each gated on green `make test`):
- HNSW: #7 real hierarchical descent; #22 reverse-edge pruning; #23 wasted descent removed; #38 promotion-level entry points; #24 M==1 ml=inf guard + level cap; #8 dup-node-on-update (key_to_node_ + node tombstones); #25 arena growth (documented tradeoff). +regression tests (dup, M=1); recall tests still pass.
- api/tcp_server: #9/#34 alloc bounds vs frame; #11/#12 send timeout + frame deadline; #10 getaddrinfo; #43 STATS saturate.
- api/tcp_client: #26 getaddrinfo; #28 server-driven alloc bounds; #35 batch OOB guard.
- api/tcp_main: #44 safe CLI parse + range checks.
- storage/mmap_storage: #4/#11 length clamps; #6 update overflow guard; #20 fd double-close; #21 file-size validation; #40 write-then-publish ordering; #42 advise clamp.
- storage/segment: #52 seal removes redundant WAL.
- features/query_cache: #3 k-aware; #19 zero-cap no-op.
- features/atomic_batch_insert: #27 size guards; #30 delete-rollback no longer resurrects with 0-dim vector.
- features/commit_log: #13 fsync per entry; #15 fixed-width filenames; #18 sequence resume.
- features/atomic_persistence: #39 payload-covering FNV CRC; #29 re-derive main_data_file_; #54 shouldCheckpoint lock; #55 load count bound + try/catch.
- optimizations/parallel_processing: #31 centroid bounds; #32 returns success count.
- optimizations/simd + utils: #33 AVX guard; #56 <cmath>; #45 dead members removed; #47 cosine via simd_ops.
- core: #2 NaN guards; #16 GPU metric fallback; checkpoint lock; #37 MMap batchInsert atomic rollback; #58 documented global SIMD.

Test count: 106 (60 unit + 26 e2e + 20 tcp), all green.

## Validation — DONE
- **Sanitizers:** full suite (106 tests) clean under **ASan+UBSan** and under **TSan** — zero errors.
  - `make test CXX="c++ -fsanitize=address,undefined -fno-omit-frame-pointer -g"`
  - `make test CXX="c++ -fsanitize=thread -O1 -g"`
- **Fuzzing:** 7 harnesses, all crash-free under ASan+UBSan via the Apple-clang fallback driver
  (`make fuzz FUZZ_CC=c++`). Real coverage-guided libFuzzer isn't available on Apple CLT
  (no `libclang_rt.fuzzer_osx.a`); harnesses are libFuzzer-compatible for coverage-guided runs elsewhere.
  - Existing: `fuzz-protocol`, `fuzz-wal` (fixed: it never called `load()`, so it wasn't parsing the WAL — now it does), `fuzz-logentry`.
  - New: `fuzz-mmap-file` (mmap file parser), `fuzz-distance` (metrics/SIMD/quantizer),
    `fuzz-sealed-segment` (HNSW snapshot parser + importGraph + rebuild fallback),
    `fuzz-db-ops` (full VectorDatabase integration, segmented engine).

## Expanded testing + fuzzing (round 2)
- **+4 e2e tests** (now 110 total: 60 unit + 30 e2e + 20 tcp), all green under release, ASan+UBSan, and TSan:
  - `segmented seal+compact+recover` — forced sealing (small mutable segment) + tombstones + compaction + cold-open recovery of exactly the live set.
  - `HNSW recall@10 large scale` — n=1500, D=16, brute-force ground truth, recall ≥ 0.75 (validates the rewritten descent + pruning; also exercises `configureHNSW`).
  - `HNSW no dup after many updates` — 40 keys × 4 update rounds, no duplicate key in any result (dup-node fix at scale, default engine).
  - `concurrent mixed ops stress` — 8 threads × 300 mixed insert/search/update/delete ops (clean under TSan).
- **+1 fuzzer** `fuzz-db-ops-mmap` — fast MMap-engine integration fuzzer (no per-op fsync) for high-throughput coverage; complements the durable segmented `fuzz-db-ops`. **8 harnesses total.**
- Longer fuzz campaign (`make fuzz FUZZ_CC=c++ FUZZ_RUN_FLAGS="-runs=50000000 -max_total_time=120"`).

## Post-audit: ANN benchmark harness + HNSW quality fix
- Added `src/utils/vecs_io.hpp` (fvecs/ivecs/bvecs loaders) + `test/bench_ann.cpp` + `make bench-ann`
  — recall-vs-QPS sweep vs exact FlatIndex ground truth (tie-aware + ID recall), CSV output.
  Runs synthetic clustered data by default; `ANN_ARGS="--data datasets/sift"` for real SIFT1M.
- **Benchmark exposed a real index-quality bug:** `selectNeighbors`/pruning used the naive
  "keep the M closest" rule → recall plateaued at ~0.36 (128d) / ~0.45 (16d) and did not respond
  to ef (graph not navigable) — the exact failure the research predicted.
- **Fix:** implemented the Malkov-Yashunin diversifying heuristic (Alg. 4) + keepPrunedConnections
  in `selectNeighbors`, and re-prune neighbor lists with the same heuristic (`pruneNeighbors`).
  Result: recall 0.36 → **1.0** (128d: 0.967@ef10 → 1.0@ef100; 960d: 0.977@ef10 → 1.0@ef64),
  monotonic with ef, peak mem sane. 110/110 tests still green.

## Research-backed optimizations (do-next 4)
Measured on synthetic clustered data (n=30k, d=128, M=16, efc=200), Arena allocator:
- **#1 arena memory** — `monotonic_buffer_resource` → `unsynchronized_pool_resource` + reserve
  adjacency to max degree. Peak HNSW memory **2046 MiB → 38 MiB (~54×)**.
- **#6 HNSW defaults** — in-memory M/efc/efs 10/8/8 → 16/200/64; segmented efc/efs 80/50 → 200/64.
- **#5 devirtualize** `getDistance` — resolve concrete metric once at ctor, call SIMD kernel directly
  (no vtable in the graph walk).
- **#4 SIMD kernels** — `squared_distance`/`dot_product` rewritten with 4 accumulators + FMA
  (NEON `vfmaq_f32`; AVX `_mm256_fmadd_ps` under `__FMA__`). #3 build flags: `-mfma` on x86,
  CMake now defaults to Release + arch flags (was silently `-O0`/scalar).
- **#2 visited set** — per-call `unordered_set` → per-thread versioned array (O(1) reset, no alloc/hash).

Net effect: **QPS@ef10 ~49k → ~97k (~2×)**, build ~6.2s → 3.9s, recall unchanged (0.97→1.0),
memory 54× lower. Validated: 110/110 tests (release + ASan+UBSan + TSan), HNSW fuzzers crash-free.

## Durability experiment + benchmark suite + quantizer fix
- **Group commit** — `VectorSegment` deferred-sync mode + `SegmentedVectorStore::insertBatch` +
  `VectorDatabase::batchInsert` now do ONE fsync per batch (was one per row). New crash-consistency
  test (`group commit crash recovery`, fork + no-shutdown) proves the batch's single fsync is durable.
- **Honest durability** — `vdb::io::set_full_fsync()` toggle: macOS `F_FULLFSYNC` (real drive-cache
  flush) vs plain `fsync`. `test/bench_durability.cpp` (+`make bench-durability`) reports the
  durability tax (fsync floor 41k/s plain vs **316/s F_FULLFSYNC**), group-commit speedup
  (**4.4× under honest durability**), and recovery (sealed 35 ms vs WAL-replay 225 ms).
- **Benchmark turn-key** — `datasets/fetch_sift.sh` (SIFT/GIST + vendors hnswlib), `test/bench_hnswlib.cpp`
  + `make bench-hnswlib` (baseline, same CSV schema). Run on a networked host.
- **Quantizer scale fix** — per-dim scale (distorted L2) → single global scale (L2-proportional);
  regression test asserts quantized ordering matches true-L2 ordering.
- Stale docs fixed: `vector_database.hpp` RCU comment corrected; README badges (111 tests, +TSan).

Test count: **111** (61 unit + 30 e2e + 20 tcp), green under release + ASan+UBSan + TSan.
Benchmark targets: `make bench-ann` (recall/QPS), `make bench-durability`, `make bench-hnswlib`.

## Status: COMPLETE (audit) + index competitive + hot paths optimized + durability experiment
All 58 review findings + build fix implemented; 110/110 tests green (release, ASan+UBSan, TSan);
8 fuzzers crash-free; HNSW recall now competitive on the ANN harness. Branch `fix/audit-findings`.
- storage/segment.cpp: #52 seal leaves wal.log; #49/#46 WAL-append dedup/efficiency.
- storage/mmap_storage.cpp: #4/#11 OOB slot lengths; #6 update metadata overflow; #20 double-close fd; #21 file-size validation; #40 flag ordering; #42 advise_willneed clamp.
- algorithms/hnsw_index.cpp: #7 descent no-op; #22 no reverse-edge prune; #23 wasted descent; #38 promotion levels; #24 M==1 ml=inf; #8 duplicate node on update; #25 arena growth.
- api/tcp_server.cpp: #9/#34 unbounded alloc DoS; #11/#12 slowloris/no send timeout; #10 inet_pton; #43 STATS u32 truncation.
- api/tcp_client.cpp: #26 inet_pton; #28 server-driven alloc; #35 batch_insert OOB.
- api/tcp_main.cpp: #44 stoi/stoul uncaught.
- features/atomic_persistence.cpp: #39 footer CRC; #29 updateConfig paths; #54 shouldCheckpoint lock; #55 loadCheckpoint validate.
- features/commit_log.cpp: #13 no fsync; #15 filename width; #18 ctor sequence resume.
- features/atomic_batch_insert.cpp: #30 delete-rollback empty vector.
- optimizations/parallel_processing.cpp: #31 centroid OOB; #32 swallowed [[nodiscard]].
- utils: #45 random_generator dead members; #47 cosine dup.

## Then
- Fuzzing: fix `make fuzz` for Apple clang (fallback driver); add harnesses for HNSW snapshot, segment files, mmap file, distance metrics, quantizer, full-DB op sequence.
- Sanitizers: run unit+e2e+tcp under ASan+UBSan and TSan; fix anything surfaced.
