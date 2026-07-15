#!/usr/bin/env bash
# On-Mac rigor runs to firm up the 7-scoring reviewers' concerns (no hardware):
#   1) snapshot-creation (sealing) cost  -- the excluded half of the recovery tradeoff
#   2) durability-tax N-sweep (10k, 30k) -- does the 4.5x ratio hold at larger N?
#   3) median-of-5 QPS on SIFT1M         -- de-noise the single-run Pareto claim
set -u
cd /Users/habib/Documents/sig/vector-database
mkdir -p docs/paper/data

echo "[rigor 1/3] sealing cost (recovery run emits seal_create_ms) $(date)"
./build/bench_durability --n 2000 --trials 3 --tag sealcost > build/rigor_sealcost.log 2>&1
cp build/durability_results/durability_plain_sealcost_d128.csv docs/paper/data/ 2>/dev/null

echo "[rigor 2/3] durability-tax N-sweep (10k, 30k; plain+full, skip-recovery) $(date)"
for N in 10000 30000; do
  ./build/bench_durability --n "$N" --trials 3 --skip-recovery --tag "n$N"           > "build/rigor_tax_n${N}_plain.log" 2>&1
  ./build/bench_durability --n "$N" --trials 3 --skip-recovery --full-fsync --tag "n$N" > "build/rigor_tax_n${N}_full.log" 2>&1
  cp build/durability_results/durability_plain_n${N}_d128.csv docs/paper/data/ 2>/dev/null
  cp build/durability_results/durability_full_n${N}_d128.csv  docs/paper/data/ 2>/dev/null
done

echo "[rigor 3/3] median-of-5 QPS on SIFT1M $(date)"
./build/bench_ann --data datasets/sift --qtrials 5 > build/rigor_median_sift.log 2>&1
cp build/ann_results/results.csv docs/paper/data/ann_sift1m_ours_median.csv 2>/dev/null

echo "[rigor] COMPLETE $(date)"; touch build/rigor.done
