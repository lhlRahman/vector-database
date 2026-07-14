# Raw result data for "Honest Durability for Graph ANN"

These CSVs are the exact numbers behind the paper's tables and figures (Apple M4
Pro, macOS/APFS, single host). They are archived here so the tables are
independently checkable; all are regenerable from source.

## Durability (Tables 1–5, Figs. 2–3) — median of 7 trials, columns: metric,d,mode,median,min,max
- `durability_plain_d128.csv` — plain `fsync`, d=128
- `durability_full_d128.csv`  — `F_FULLFSYNC`, d=128

Regenerate: `make bench-durability` (add `DURABILITY_ARGS="--full-fsync"` for the
`F_FULLFSYNC` mode; `scripts/run_durability_all.sh` runs both modes at d=64,128).
Figures: `scripts/plot_durability.py`.

The d=64 dimension-robustness numbers quoted inline in §4 come from the same
harness with `--d 64`.

## Index competitiveness (Table 6, Fig. 1) — SIFT1M, single-thread
- `ann_sift1m_ours.csv`     — this work (ID-set recall@10 vs shipped ground truth)
- `ann_sift1m_hnswlib.csv`  — hnswlib baseline, identical dataset/params/host

Regenerate: `make bench-ann ANN_ARGS="--data datasets/sift"` and
`make bench-hnswlib HNSWLIB_ARGS="--data datasets/sift"`. Figure: `scripts/plot_ann.py`.
QPS is single-run timing; recall is deterministic at fixed parameters.
