# Raw result data for "Honest Durability for Graph ANN"

These CSVs archive the inputs behind the paper's tables and figures. The legacy
durability study spans Apple and AWS Linux; the committer results use one Apple
M4 Pro. Provenance and regeneration commands are separated below.

## Legacy durability study
- `durability_plain_d128.csv` — plain `fsync`, d=128
- `durability_full_d128.csv`  — `F_FULLFSYNC`, d=128

Regenerate: `make bench-durability` (add `DURABILITY_ARGS="--full-fsync"` for the
`F_FULLFSYNC` mode; `scripts/run_durability_all.sh` runs both modes at d=64,128).
Figures: `scripts/plot_durability.py`.

## Index competitiveness (SIFT1M, single-thread)
- `ann_sift1m_ours.csv`     — this work (ID-set recall@10 vs shipped ground truth)
- `ann_sift1m_hnswlib.csv`  — hnswlib baseline, identical dataset/params/host

Regenerate: `make bench-ann ANN_ARGS="--data datasets/sift"` and
`make bench-hnswlib HNSWLIB_ARGS="--data datasets/sift"`. Figure: `scripts/plot_ann.py`.
QPS is single-run timing; recall is fixed only for the archived graph build
(independent builds use randomized HNSW levels).

## Real recall committer

The committer paper results are intentionally split by provenance:

- `recall_committer_reported_run.csv` preserves the selected aggregate columns
  printed by the designated milestone run. It is the sole source for the
  throughput, ACK-latency, `max_W`, synchronization-count, and alarm ranges in
  Table `tab:commit`.
- `recall_committer_crash.csv` contains the 192 recovered-query observations
  (eight cases, two repetitions, 12 queries). It is the source for `M`, `L`,
  positive recall delta, amplification, lost-ID containment, and stable-candidate
  fingerprint claims. The file was byte-identical across benchmark reruns.
- `recall_committer_summary.csv` is deterministically regenerated from those two
  inputs by `../plot_recall_committer.py`; its confidence intervals use 10,000
  fixed-seed bootstrap resamples over the 24 correlated query observations per
  displayed case. The LaTeX/PGFPlots figure reads this CSV directly.
- `recall_committer.csv`, `recall_committer_operations.csv`, and
  `recall_committer_run.txt` are a later timing rerun. They reproduce correctness
  but not the designated performance ranges and are not the source for reported
  throughput.
- `committer_unit_test.txt`, `committer_crash_test.txt`, and
  `committer_cut_test.txt` archive the 22 committer tests (plus 72 legacy tests),
  the 15/15 real `_exit` matrix, and the 661/661 WAL-prefix result.
- `recall_committer_environment.txt` records the Apple M4 Pro/macOS/clang
  environment. The Linux `dm-log-writes` committer harness is implemented but has
  not been run; there is therefore no physical replay result in this directory.

The benchmark defaults are 160 clustered base records, 25 writes, 32 concurrent
queries, `d=12`, `k=10`, four writers, two repetitions, `ef=32`, and seed 100.
The 192 rows represent 12 recovery queries over each of 16 cloned live images,
not 192 process crashes. Regenerate the portable evidence from the repository
root with:

```sh
make committer-unit-test
make committer-crash-test
make committer-cut-test
make bench-recall-committer \
  RECALL_COMMIT_ARGS="--output build/ann_results"
(cd docs/paper && python3 plot_recall_committer.py)
```
