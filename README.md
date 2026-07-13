# vector-database

A from-scratch C++20 vector similarity-search engine. Exact (flat SIMD) and
approximate (HNSW) nearest-neighbor search, default write-ahead-logged
segmented storage with crash-safe recovery, binary TCP protocol.

> **What this is:** a learning / portfolio project built to understand the
> internals of a vector database end to end — index, storage, recovery,
> protocol, concurrency, fuzzing.
>
> **What this is not:** a FAISS / hnswlib / Qdrant replacement. Use those
> for production. Read this for the implementation.

`tests: 111 passing` · `sanitizers: ASan + UBSan + TSan clean` · `fuzzed: 8 harnesses, crash-free`

## Quick start

```sh
git clone https://github.com/lhlrahman/vector-database
cd vector-database
make tcp-server          # binary at build/tcp_server
./build/tcp_server --dims 128 --host 127.0.0.1 --port 9090 --threads 4
```

Embed in C++:

```cpp
#include "core/vector_database.hpp"

int main() {
    VectorDatabase db(/*dimensions=*/128);
    db.initialize();

    std::vector<float> v(128, 0.5f);
    [[maybe_unused]] bool ok = db.insert(Vector(std::move(v)), "key-0", "metadata");

    Vector query(std::vector<float>(128, 0.5f));
    for (const auto& [key, distance] : db.similaritySearch(query, /*k=*/10)) {
        std::cout << key << "  " << distance << "\n";
    }
    db.shutdown();
}
```

State location: by default `VectorDatabase` uses the segmented engine and
creates an auto-cleaned temp directory under `/tmp/vdb_auto_<pid>_<time>`.
Pass an explicit `storage_path` to keep data across process restarts. The
legacy mmap engine can still be selected explicitly with
`StorageEngine::MMap`.

## Features

- **HNSW** approximate NN with arena-allocated graph nodes; configurable
  `M` / `ef_construction` / `ef_search`.
- **FlatIndex** — exact brute force, templated on a `RawDistanceMetric`
  policy (Euclidean / Manhattan / cosine).
- **SegmentedVectorStore** — WAL-backed mutable segments + sealed HNSW
  snapshots, online compaction, recovery via snapshot rather than WAL
  replay. Single insert/update/delete operations are ACID-compliant: each
  WAL or tombstone append is flushed with `fsync` before the call returns,
  and every rename is followed by `fsync` on file *and* parent dir.
- **MMapStorage** — slot-based memory-mapped store, zero-copy reads via
  `std::span<const float>`.
- **SIMD** distance kernels — ARM NEON and x86 AVX2 (squared L2,
  Manhattan, dot product), scalar fallback.
- **Scalar quantization** (float32 → uint8) for fast candidate filtering
  before exact re-rank.
- **Binary TCP protocol** — fixed 7-byte header, little-endian
  length-prefixed frames, capped at 64 MiB. No HTTP/JSON.
- **Concurrency** — single `RWLock` (std::shared_mutex), shared reads
  and exclusive writes.
- **Query cache** — LRU on query hash, generation-counter invalidated
  on every mutation.

## How it works

A `VectorDatabase` composes:

| Layer    | Implementation                                                                         |
| -------- | -------------------------------------------------------------------------------------- |
| Storage  | `SegmentedVectorStore` by default (WAL + sealed snapshots), or explicit `MMapStorage` |
| Index    | `FlatIndex<MetricPolicy>` (exact) **or** `HNSWIndex` (approximate)                     |
| Cache    | `QueryCache` (LRU, generation-invalidated)                                             |
| Locking  | `RWLock` — `std::shared_mutex` for readers vs writer exclusion                         |
| Metrics  | `EuclideanDistance`, `ManhattanDistance`, `CosineSimilarity`                           |

Indexes never copy vector bytes — they read through a `VectorAccessor`
callback that returns a pointer into the underlying storage.

`SegmentedVectorStore` follows a Qdrant-style lifecycle: writes go into a
mutable segment with WAL; once a size or tombstone-ratio threshold is
crossed the segment is sealed (HNSW snapshot written), and sealed
segments are compacted in the background. Cold opens load HNSW from the
snapshot files instead of replaying the WAL — that's where the recovery
speedup comes from.

## Building

Requires a C++20 compiler (gcc 10+, clang 12+; `make fuzz` needs clang
17+ for libFuzzer). POSIX-y system. No external dependencies.

```sh
make tcp-server      # server binary
make test            # unit + e2e + tcp tests (100 tests)
make fuzz            # libFuzzer harnesses (uses clang if installed,
                     # falls back to a gcc random-mutation driver)
```

With sanitizers:

```sh
CXX="g++ -fsanitize=address,undefined -fno-omit-frame-pointer -g" make test
CXX="g++ -fsanitize=thread -O1 -g"                                make test
```

Status as of the latest local run on Linux/aarch64: 100/100 release tests
pass; unit + e2e pass under ASan+UBSan; 40.2 M libFuzzer iterations across
three harnesses with zero crashes.

The macOS Metal GPU code path exists in `src/optimizations/` but is not
exercised by CI on this machine; the Linux build always links the no-op
GPU stub.

## Performance

All numbers below are from a single host (gcc 15, `-O2`, release build,
no sanitizers, Linux/aarch64 on Apple Silicon under virtualization).
Point-in-time measurements, not promises. **`make perf-test` to reproduce.**
The primary API tables use the default segmented engine. On this host
`/tmp` is tmpfs, so `make perf-test` shows the segmented code path without
real persistent-disk flush latency; the persistence table below shows the
cost on btrfs/NVMe.

**Single-thread default API** (segmented engine, n=1000, d=128 unless noted):

| Operation                           | Throughput  |
| ----------------------------------- | ----------- |
| insert (d=32, single)               |  33 K ops/s |
| insert (d=128, single)              |  27 K ops/s |
| batch insert (d=32, batch of 5000)  |  34 K ops/s |
| update                              |  29 K ops/s |
| delete                              | 257 K ops/s |
| segmented search (Exact flag, d=32) |  45 K qps   |
| segmented search (HNSW flag, d=32)  |  35 K qps   |
| segmented search (n=5000, d=64)     |  17 K qps   |
| query-cache hit                     | 3.2 M ops/s |

With segmented storage, the `SearchMode` enum is not yet wired through to
choose flat exact search; segmented search uses per-segment HNSW today.

**Concurrent search** (n=1000, d=32, segmented engine):

| Threads | Throughput |
| ------- | ---------- |
| 1       |  12 K qps  |
| 4       |  42 K qps  |
| 8       |  75 K qps  |

The `RWLock` lets readers run in parallel, so throughput scales well up
to 8 threads in this small benchmark.

**Segmented search latency vs dimensions** (n=1000):

| Dimensions | Per query | qps   |
| ---------- | --------- | ----- |
|   8        | 24.4 µs   | 41 K  |
|  32        | 29.0 µs   | 34 K  |
|  64        | 35.3 µs   | 28 K  |
| 128        | 52.6 µs   | 19 K  |
| 256        | 72.4 µs   | 14 K  |

(The single-thread d=32 number here differs from the concurrent table
above because the two benchmarks use different query batches and warmup;
each benchmark is self-consistent.)

**TCP transport overhead** (`make bench-tcp`, default segmented engine,
d=128, loopback). Inserts are dominated by segmented WAL writes, so TCP
adds only a few microseconds there. Read/search overhead is mostly the
syscall/framing cost:

| Op                          | Direct call    | TCP framed    | Per-call cost added |
| --------------------------- | -------------- | ------------- | ------------------- |
| Insert (1000)               | 18 K ops/s     | 17 K ops/s    | ~3 µs               |
| Search top-10 (1000)        | 545 K ops/s    | 107 K ops/s   | ~8 µs               |
| Get by key (2000)           | 8.8 M ops/s    | 30 K ops/s    | ~34 µs              |
| Concurrent search (4×500)   | 333 K ops/s    | 73 K ops/s    | ~11 µs              |

Wire-size: a 128-dim search request is 527 bytes binary vs ~1 352 bytes
HTTP+JSON (61 % reduction).

**Segmented vs legacy persistence** (`make bench-segmented-persistence`,
n=5000, d=64, plus 250 deletes + 200 updates). The segmented engine
fsyncs the WAL on every mutation — the cost is real and you have to pay
it on a real filesystem to see the truth:

| | mmap-monolith | segmented (tmpfs) | segmented (btrfs/NVMe) |
| --- | --- | --- | --- |
| Insert avg | 254 µs | 94 µs | **319 µs** |
| Insert p99 | — | 166 µs | 653 µs |
| Insert max | — | 3.5 ms | 35.0 ms |
| Update avg | 415 µs | 28 µs | 254 µs |
| Delete avg | 0.2 µs | 5 µs | 157 µs |
| Search avg (post-compact) | 337 µs | 313 µs | 357 µs |
| **Recovery (cold open)** | **1170 ms** | 11 ms | **13 ms** |
| Disk usage | 6.1 MiB | 5.1 MiB | 5.1 MiB |

Two things to read out of this:

1. **The tmpfs column was a lie.** Earlier benchmarks ran under `/tmp`
   which is RAM-backed on most Linux distros — `fsync()` is a no-op there.
   Run with `TMPDIR=/path/on/persistent/disk` to get honest numbers.
2. **Per-write fsync hurts.** Average insert latency is close to mmap on
   this run, but deletes are hundreds of times slower because the mmap
   path only updates process-local/page-cache state. The segmented engine
   also has p99 latency in the sub-ms range and 35 ms tail spikes when the
   filesystem decides to flush a transaction group. The price of true
   durability. Group commit / batched fsync would amortize it but isn't
   implemented yet.

The mmap engine's "fast" insert (234 µs) is a different number entirely
— it's not durable on power loss, the writes go to the page cache and
the kernel flushes when it feels like it. **It is wrong to compare those
numbers as if they measure the same thing.** The segmented engine
guarantees the call doesn't return until the WAL record is on disk; the
mmap engine guarantees nothing.

Recovery: ~90× faster on segmented because sealed segments load HNSW
from a snapshot file instead of replaying every insert through the
graph builder.

For the full configuration matrix and tradeoff notes, see
[`docs/benchmarks.md`](docs/benchmarks.md).

**Fuzzing.** Three libFuzzer harnesses (protocol parser, WAL recovery,
LogEntry deserializer) ran for 40.2 M iterations under ASan+UBSan with
zero crashes.

**HNSW arena allocator.** `make bench-hnsw-allocator` validates that
swapping `std::allocator` for a `std::pmr::monotonic_buffer_resource`
arena reduces allocation calls **17 443×** while keeping recall identical
in the current benchmark. Arena search was slightly faster in this run
(314 vs 324 µs/query), at the cost of higher reserved peak memory.

## Project layout

```
src/
  core/          VectorDatabase, Vector
  algorithms/    HNSWIndex, FlatIndex
  storage/       MMapStorage, VectorSegment, SegmentedVectorStore
  api/           TCP server, client, binary protocol
  features/      QueryCache, AtomicPersistence, CommitLog, batching
  optimizations/ SIMD kernels, RWLock, scalar quantization, parallel ops
  utils/         distance metrics, atomic_write helper, RNG
test/
  unit_tests.cpp / e2e_tests.cpp / test_tcp.cpp
  fuzz_*.cpp                 (libFuzzer harnesses + corpus generator)
  bench_*.cpp                (microbenchmarks)
```

## API reference

Selected public methods on `VectorDatabase` (full surface in
`src/core/vector_database.hpp`):

```cpp
// Lifecycle
void initialize();
void shutdown();

// Mutations — all [[nodiscard]]
bool insert(const Vector& v, const std::string& key,
            const std::string& metadata = "");
bool update(const Vector& v, const std::string& key,
            const std::string& metadata = "");
bool remove(const std::string& key);

// Reads
std::optional<Vector>                                   get(const std::string& key);
std::vector<std::pair<std::string, float>>              similaritySearch(const Vector&, size_t k);
std::vector<SearchResult>                               similaritySearchWithMetadata(const Vector&, size_t k);

// Batching
BatchResult batchInsert(const std::vector<std::string>& keys,
                        const std::vector<Vector>&      vectors,
                        const std::vector<std::string>& metadata = {});
BatchResult batchDelete(const std::vector<std::string>& keys);

// Configuration
void setSearchMode(SearchMode);                         // Exact | HNSW
void setDistanceMetric(std::shared_ptr<DistanceMetric>);
void configureHNSW(size_t M, size_t ef_construction, size_t ef_search);
void configureSegmentedStorage(size_t max_mutable_records,
                               size_t max_sealed_segments,
                               double max_tombstone_ratio);

// Persistence
bool checkpoint();
void sealMutableSegment();
void compactSegments();

// Stats
Statistics getStatistics() const;
size_t     vectorCount() const;
```

Wire protocol: see comments at the top of `src/api/protocol.hpp`.

## License

[MIT](LICENSE).
