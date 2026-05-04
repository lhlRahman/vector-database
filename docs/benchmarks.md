# Benchmark Matrix

This document records the supported benchmark configurations and the tradeoffs
between them. It is meant to be exhaustive over the configuration surface, not
over every possible numeric parameter value.

Numbers are point-in-time measurements from one local host:

| Field | Value |
| --- | --- |
| CPU/OS | Linux/aarch64 on Apple Silicon under virtualization |
| Compiler | g++ 15.2.1, `-O2`, no sanitizers |
| Date | 2026-05-04 |
| Default storage | `StorageEngine::Segmented` |
| `/tmp` filesystem | tmpfs |
| repo filesystem | btrfs/NVMe |

Use these commands to reproduce the current benchmark set:

```sh
make perf-test
make bench-tcp
make bench-segmented-persistence
TMPDIR=/path/on/persistent/disk make bench-segmented-persistence
make bench-hnsw-allocator
```

The benchmark programs do not currently sweep every knob automatically. Tables
below mark whether a row is directly measured by an existing harness or is a
configuration tradeoff inferred from the implementation.

## Configuration Axes

| Axis | Values | Where it is set | Notes |
| --- | --- | --- | --- |
| Storage engine | `Segmented`, `MMap` | `VectorDatabase(..., storage_engine)` | Segmented is the default. MMap is legacy and must be selected explicitly. |
| Search mode | `Exact`, `HNSW` | constructor or `setSearchMode()` | Fully active for MMap. Segmented currently searches per-segment HNSW regardless of the enum. |
| Distance metric | Euclidean, Manhattan, cosine, custom `DistanceMetric` | `setDistanceMetric()` | Exact MMap dispatches optimized policy paths for built-ins. HNSW/segmented use the virtual metric object. |
| HNSW quality | `M`, `ef_construction`, `ef_search` | `configureHNSW()` | Higher values improve recall and cost more CPU/memory. |
| HNSW allocator | `Standard`, `Arena` | `configureHNSWAllocator()` | Arena reduces allocation calls dramatically and is the segmented default. |
| Segment layout | mutable record limit, sealed segment limit, tombstone ratio | `configureSegmentedStorage()` | Controls sealing/compaction frequency and read fanout. |
| Query cache | enabled/disabled, capacity | constructor | LRU, generation-invalidated on mutation. Best for repeated identical queries. |
| Batch ops | enabled/disabled | constructor | Required for `batchInsert`, `batchUpdate`, `batchDelete`. |
| Atomic persistence feature | enabled/disabled | constructor | Separate feature layer around commit-log/checkpoint support; segmented storage has its own WAL. |
| SIMD | enabled/disabled | `enableSIMD()` | Affects distance kernels where supported by the platform. |
| GPU | enabled/disabled, threshold | `enableGPU()`, `setGPUThreshold()` | Metal path exists on macOS; Linux links a no-op stub. Not part of current Linux benchmarks. |
| Transport | direct C++ API, TCP binary protocol | benchmark or server/client | TCP adds syscall/framing cost but avoids HTTP/JSON overhead. |
| Filesystem | tmpfs, persistent disk | `TMPDIR` or explicit path | Critical for segmented write latency because each mutation fsyncs the WAL. |

## Storage Engine Tradeoffs

| Engine | Durability | Recovery | Write cost | Read/search shape | Best use |
| --- | --- | --- | --- | --- | --- |
| `Segmented` | Single insert/update/delete operations are ACID-compliant. Each WAL/tombstone append is fsynced before returning. | Fast cold opens from sealed HNSW snapshots plus manifest. | Higher tail latency on persistent filesystems because fsync is on the critical path. | Searches mutable and sealed segments, merges per-segment candidates. | Default, crash-safe local persistence, realistic storage behavior. |
| `MMap` | Not durable on power loss unless the caller explicitly syncs and the OS flushes. | Rebuilds index from the mmap file. | Fast apparent writes because they mostly hit page cache/process state. | Exact flat scan or one HNSW index over mmap slots. | Legacy comparisons, fast experiments where durability does not matter. |

Measured by `make bench-segmented-persistence`, `n=5000`, `d=64`,
250 deletes, 200 updates, `k=10`.

| Metric | mmap-monolith | segmented on tmpfs | segmented on btrfs/NVMe |
| --- | ---: | ---: | ---: |
| Insert avg | 254 us | 94 us | 319 us |
| Insert p99 | n/a | 166 us | 653 us |
| Insert max | n/a | 3.5 ms | 35.0 ms |
| Update avg | 415 us | 28 us | 254 us |
| Delete avg | 0.2 us | 5 us | 157 us |
| Search avg after compact | 337 us | 313 us | 357 us |
| Cold recovery | 1170 ms | 11 ms | 13 ms |
| Disk usage | 6.1 MiB | 5.1 MiB | 5.1 MiB |

The tmpfs column is useful for CPU-path comparisons but not for durability
latency. On tmpfs, fsync does not represent physical media flush cost.

## Search Configurations

| Storage | Search mode | Actual behavior today | Tradeoff |
| --- | --- | --- | --- |
| `MMap` | `Exact` | Flat scan over all live slots. | Deterministic recall, O(n) query cost, no graph memory. |
| `MMap` | `HNSW` | Single HNSW graph over mmap slot IDs. | Faster large-n search at approximate recall; writes maintain graph. |
| `Segmented` | `Exact` | Per-segment HNSW search despite the enum. | Known limitation; do not use this as exact ground truth. |
| `Segmented` | `HNSW` | Per-segment HNSW search and candidate merge. | Intended segmented search path; query cost grows with segment fanout. |

Measured by `make perf-test` with default segmented storage:

| Operation | Throughput |
| --- | ---: |
| segmented search, `Exact` flag, n=1000, d=32, k=10 | 45 K qps |
| segmented search, `HNSW` flag, n=1000, d=32, k=10 | 35 K qps |
| segmented search, n=5000, d=64, k=10 | 17 K qps |
| query-cache hit | 3.2 M ops/s |

Dimension sweep from the same harness, `n=1000`, 200 queries:

| Dimensions | Avg/query | Throughput |
| ---: | ---: | ---: |
| 8 | 24.4 us | 41 K qps |
| 32 | 29.0 us | 34 K qps |
| 64 | 35.3 us | 28 K qps |
| 128 | 52.6 us | 19 K qps |
| 256 | 72.4 us | 14 K qps |

Size sweep, `d=32`, 200 queries:

| Vectors | Avg/query | Throughput |
| ---: | ---: | ---: |
| 100 | 9.3 us | 107 K qps |
| 500 | 21.7 us | 46 K qps |
| 1000 | 29.8 us | 34 K qps |
| 5000 | 51.0 us | 20 K qps |

## Write Configurations

Measured by `make perf-test` with default segmented storage on tmpfs:

| Operation | Throughput | Avg/op |
| --- | ---: | ---: |
| insert, 1000 vectors, d=32 | 33 K ops/s | 30.7 us |
| insert, 5000 vectors, d=32 | 25 K ops/s | 40.7 us |
| insert, 1000 vectors, d=128 | 27 K ops/s | 36.5 us |
| batch insert, 1000 vectors, d=32 | 45 K ops/s | 22.0 us |
| batch insert, 5000 vectors, d=32 | 34 K ops/s | 29.7 us |
| update, 500 vectors, n=1000 | 29 K ops/s | 34.8 us |
| delete, 1000 vectors | 257 K ops/s | 3.9 us |

Important interpretation: these write numbers use `/tmp` auto paths, which are
tmpfs on this machine. Use `TMPDIR=/path/on/persistent/disk` or an explicit
storage path for real fsync costs.

## HNSW Knobs

| Parameter | Higher value does | Cost |
| --- | --- | --- |
| `M` | Keeps more graph neighbors per node. Usually improves recall and routing quality. | More memory, more insert/search work. |
| `ef_construction` | Explores more candidates while building. Improves graph quality. | Slower inserts/builds. |
| `ef_search` | Explores more candidates while querying. Improves recall. | Slower searches. |

Current defaults differ by layer:

| Layer | `M` | `ef_construction` | `ef_search` | Allocator |
| --- | ---: | ---: | ---: | --- |
| MMap `VectorDatabase` HNSW defaults | 10 | 8 | 8 | Standard |
| `SegmentedVectorStore::Config` defaults | 16 | 80 | 50 | Arena |

Measured by `make bench-hnsw-allocator`, `n=5000`, `d=64`, `queries=500`,
`k=10`, `M=16`, `ef_construction=80`, `ef_search=50`:

| Allocator | Build | Search | Avg/query | QPS | Recall@10 | Allocation calls | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 1271 ms | 162 ms | 324 us | 3085 | 0.985 | 87215 | 6.72 MiB |
| Arena | 1211 ms | 157 ms | 314 us | 3182 | 0.985 | 5 | 13.19 MiB |

Arena is primarily a latency/allocation-count improvement. It uses a larger
reserved memory arena in this benchmark.

## Segmentation Knobs

| Setting | Lower value | Higher value |
| --- | --- | --- |
| `max_mutable_segment_records` | Seals more often; more sealed segments; lower mutable WAL replay at seal time. | Seals less often; larger mutable segment; fewer segment transitions. |
| `max_sealed_segments` | Compacts sooner; more compaction work but lower search fanout. | Delays compaction; cheaper writes in the short term but more segments to search. |
| `max_tombstone_ratio` | Compacts sooner after deletes/updates. | Tolerates more stale records and tombstones. |

Tradeoff summary:

| Goal | Prefer |
| --- | --- |
| Lowest search latency | Fewer sealed segments, lower tombstone ratio, periodic compaction. |
| Lowest write interruption | Larger mutable segments and higher sealed segment limit. |
| Fast recovery | Sealed segments with snapshots; avoid one giant unsealed mutable segment. |
| Lower disk amplification after churn | More aggressive compaction. |

## Distance Metrics

| Metric | Exact MMap path | HNSW/segmented path | Tradeoff |
| --- | --- | --- | --- |
| Euclidean | Optimized policy/SIMD path. | Supported. | Default and best-covered benchmark path. |
| Manhattan | Optimized policy/SIMD path. | Supported through metric object. | Good for L1-style distance; less benchmark coverage. |
| Cosine | Optimized policy path. | Supported through metric object. | Useful for normalized embeddings; verify scoring semantics for your data. |
| Custom `DistanceMetric` | Virtual fallback. | Supported through metric object. | Flexible but slower and not SIMD-specialized. |

Changing the metric clears the query cache and rebuilds index state where
needed.

## Cache Configurations

| Config | Behavior | Best use | Risk |
| --- | --- | --- | --- |
| cache enabled | LRU cache keyed by query hash; invalidated by generation on mutation. | Repeated identical queries on mostly-read workloads. | Extra memory; no benefit for unique queries. |
| cache disabled | Every query hits the index/storage path. | Write-heavy or high-cardinality query workloads. | Lower repeated-query throughput. |

Measured by `make perf-test`, `n=1000`, `d=32`, 1000 repeated queries:

| Mode | Throughput | Avg/query |
| --- | ---: | ---: |
| Cache hit | 3.2 M ops/s | 0.3 us |
| No cache | 51 K ops/s | 19.5 us |

## Batch And Atomic Persistence Features

| Feature | Applies to | Behavior | Tradeoff |
| --- | --- | --- | --- |
| Batch ops enabled | `batchInsert`, `batchUpdate`, `batchDelete` | Enables batch API entry points and batch result accounting. | Must be opted in; batches still execute through storage/index mutation paths. |
| Batch ops disabled | default constructor behavior | Batch API throws if called. | Smaller surface for simple usage. |
| Atomic persistence enabled | legacy feature layer | Initializes `AtomicPersistence` and commit/checkpoint helpers. | Separate from segmented WAL durability; use only when testing that feature. |
| Atomic persistence disabled | default | No extra commit-log/checkpoint layer. | Segmented still has durable single-op WAL. |

Do not treat batch throughput as equivalent to group commit. The segmented
engine currently fsyncs single mutations; group commit/batched fsync is not
implemented.

## SIMD And GPU

| Config | Current Linux benchmark status | Tradeoff |
| --- | --- | --- |
| SIMD enabled | Default platform kernels are compiled. | Faster exact distance kernels where the code path uses them. |
| SIMD disabled | Supported by API. | Useful for debugging/comparison, slower for flat scans. |
| GPU enabled on Linux | No-op stub is linked. | No acceleration in current Linux runs. |
| GPU enabled on macOS | Metal implementation exists but is not covered by current CI/benchmarks. | Potentially useful for large flat/batch distance work; needs platform validation. |

## TCP Transport

Measured by `make bench-tcp`, default segmented engine, `d=128`, loopback:

| Operation | Direct API | TCP framed | Per-call cost added |
| --- | ---: | ---: | ---: |
| Insert, 1000 vectors | 18 K ops/s | 17 K ops/s | about 3 us |
| Search top-10, 1000 queries | 545 K ops/s | 107 K ops/s | about 8 us |
| Get by key, 2000 lookups | 8.8 M ops/s | 30 K ops/s | about 34 us |
| Concurrent search, 4x500 | 333 K ops/s | 73 K ops/s | about 11 us |

The binary request for a 128-dimensional search is 527 bytes versus roughly
1352 bytes for an HTTP+JSON equivalent in the benchmark estimate.

## Recommended Configurations

| Scenario | Configuration | Why |
| --- | --- | --- |
| Default local persistent DB | `StorageEngine::Segmented`, explicit storage path, cache enabled. | Durable single operations and fast snapshot recovery. |
| Benchmarking raw exact scan behavior | `StorageEngine::MMap`, `SearchMode::Exact`, cache disabled. | Avoids segmented HNSW and cache effects. |
| Approximate search comparison | `StorageEngine::MMap`, `SearchMode::HNSW`, tuned HNSW params. | Single graph, clearer recall comparisons. |
| Write-heavy durable workload | Segmented with larger mutable segments; consider persistent disk tests. | Reduces sealing churn, but fsync remains per mutation. |
| Read-heavy durable workload | Segmented, compacted sealed segments, cache enabled. | Lower search fanout and cheap repeated queries. |
| Recovery-sensitive workload | Segmented with periodic sealing. | Cold opens load HNSW snapshots rather than rebuilding all inserts. |
| Low-memory graph comparison | HNSW standard allocator. | Lower peak in allocator benchmark, many more allocation calls. |
| Lower allocation overhead | HNSW arena allocator. | Far fewer allocations and slightly faster build/search in current benchmark. |

## Gaps In Current Benchmarks

The current harnesses do not yet provide an automatic full Cartesian product
over all configuration axes. Missing useful sweeps include:

- MMap exact vs MMap HNSW through `VectorDatabase` after the default switch.
- HNSW recall/latency curves across `M`, `ef_construction`, and `ef_search`.
- Segmentation threshold sweeps for mutable size, sealed segment limit, and
  tombstone ratio.
- Euclidean vs Manhattan vs cosine performance across exact and HNSW paths.
- Cache capacity sensitivity.
- Persistent-disk `make perf-test` equivalent with explicit storage paths.
- macOS Metal GPU measurements.

Those should be added as separate benchmark harnesses before publishing claims
for the missing combinations.
