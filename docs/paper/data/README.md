# Raw data for "Recall-Bounded Durability for Graph-Based ANN Indexes"

These tracked files are the inputs behind the paper's tables and figures. The
legacy durability study spans Apple and AWS Linux; the committer results use one
Apple M4 Pro. Commands below are run from the repository root unless noted.

## Legacy pre-committer durability study

- `durability_plain_d128.csv` and `durability_full_d128.csv`: Apple plain
  `fsync` and `F_FULLFSYNC`, `d=128`.
- `durability_linux_plain_d128.csv`: AWS Linux/ext4/NVMe cross-check.
- `durability_plain_nsweep_d128.csv`: Apple recovery-size sweep.
- `durability_{plain,full}_n{10000,30000}_d128.csv`: Apple tax scale checks.
- `durability_plain_sealcost_d128.csv`: legacy mutable-HNSW sealing costs.

These tax, group-commit, and recovery files predate commit `4017fa7`, when the
mutable segment inserted directly into HNSW. The current engine exact-scans a raw
mutable delta and builds HNSW only at seal, so current `make bench-durability`
does not reproduce the same index work. The CSVs remain the archived evidence
for the explicitly labeled pre-committer baseline.

`legacy_durability_sha256.txt` authenticates all 11 archived durability tables.
`legacy_durability_provenance.txt` pins their first archived revisions and the
best available source-harness revision, and records the missing raw trials,
commands, host details, and exact-producing-tree limitations.

## Index competitiveness (SIFT1M, single-thread)

- `ann_sift1m_ours.csv`: this work, ID-set recall@10 against shipped truth.
- `ann_sift1m_hnswlib.csv`: hnswlib baseline, same data/parameters/host.

Regenerate with `make bench-ann ANN_ARGS="--data datasets/sift"` and
`make bench-hnswlib HNSWLIB_ARGS="--data datasets/sift"`. The baseline is
hnswlib v0.9.0 commit `d9b3608c83d83b46c96e25088cb1d729b29dcfe9`.
`datasets/fetch_sift.sh` pins that revision and verifies these SHA-256 values:

```text
92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a  sift.tar.gz
21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816  sift_base.fvecs
f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc  sift_query.fvecs
2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f  sift_groundtruth.ivecs
331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea  sift_learn.fvecs
```

QPS is single-run timing. Recall is fixed only for each archived graph build;
independent HNSW builds may differ because level selection is randomized.

## Legacy recall-committer archive

The v1 evidence is preserved at
`recall_committer_runs/20260719-legacy-v1-73ea1bf`, from source commit
`73ea1bfa90dd38840bc6bc65e28dc2bfe4f439a6`. It contains 240 aggregate images,
2,880 recovered-query observations, 13,920 operations, and its original test,
environment, summary, and digest files. This version is retained for provenance;
it is not a top-level bundle and is not used by the current paper.

Audit the immutable legacy version explicitly with:

```sh
scripts/run_recall_committer_evidence.sh \
  --repetitions 30 \
  --validate-existing \
    docs/paper/data/recall_committer_runs/20260719-legacy-v1-73ea1bf \
  --no-install
```

## Canonical recall-committer evidence

`recall_committer_canonical` points to
`recall_committer_runs/20260719-round3-ea62d9f`. The canonical v2 bundle was
produced from source commit
`ea62d9fecfde4539070d6667f3c28e7320b1f34e` and is the current paper evidence.
It contains:

- 240 aggregate images: 30 executions for each of eight cases.
- 2,880 recovered-query observations: 12 per image.
- 14,040 write, query, fence, crash-cohort, and resumed-suffix operations.
- 60 exposed weak records, all 60 recovered, with 0 lost.
- 90 `L < M` query observations across all 30 strict-hot images.
- 30 stable post-recovery suffixes verified through a second recovery.
- 76 general tests, 24 committer tests, 15/15 process-crash frontiers, and
  661/661 WAL byte cuts.

Schema v2 alternates two fixed HNSW base images (seeds 100 and 117) 15 times per
case. Each strict-hot image exits at the declared
`fence-after-sync-before-publish` frontier, recovers both exposed weak records,
and then appends its verified stable suffix. The other seven cases retain the
terminal-suffix control.

The statistical unit is one tail/write live image. Recovery queries are
averaged within an image; summaries report min, median, and max across images
without bootstrapping correlated query rows. The query set and two base graphs
remain fixed, so repetitions are not independent graph builds. Fixed-count and
fixed-time controls remain in raw invariant totals but are omitted from the six
displayed summary rows.

Run a new portable evidence bundle without rebinding the paper's tracked
canonical evidence with:

```sh
scripts/run_recall_committer_evidence.sh --repetitions 30 \
  --output build/recall-committer-evidence
```

The runner requires clean relevant sources, snapshots HEAD and a complete source
hash manifest, runs `make clean`, and verifies the commit, cleanliness, and
source hashes again after execution. It stages all test and benchmark logs, raw
CSVs, environment, regenerated summary, and SHA-256 manifest outside `build/`,
then validates exact schemas, counts, seeds, and recovery invariants. Files and
directories are synced before a same-filesystem rename publishes the complete
version, and parent directories are synced around the atomic canonical-symlink
replacement. Unrelated and legacy data are not rewritten.

Audit the canonical bundle without running or installing evidence with:

```sh
scripts/run_recall_committer_evidence.sh \
  --repetitions 30 \
  --validate-existing docs/paper/data/recall_committer_canonical \
  --no-install
```

An audit resolves the canonical link once, verifies the original manifest, and
independently regenerates the summary and manifest for byte-for-byte comparison.
The paper should be compiled only after this tracked-canonical audit; a fresh
timing run is deliberately installed under `build/` and is not silently mixed
with the archived numbers in the prose and tables.
The runner exercises logical crash images and production read-only recovery; it
never invokes `scripts/powerloss_committer.sh`. The Linux `dm-log-writes`
harness has not been executed, so no physical block-replay result is present.
