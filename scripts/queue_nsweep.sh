#!/bin/bash
# Lever 3: recovery-time N-sweep to larger scale. Waits for the recall-staleness
# SIFT run to finish first (no CPU contention), then sweeps recovery N up to 1e5.
set -u
cd /Users/habib/Documents/sig/vector-database
echo "[nsweep] waiting for recall-staleness SIFT to finish ... $(date)"
while pgrep -f 'bench_recall_staleness --data datasets/sift' >/dev/null 2>&1; do sleep 20; done
echo "[nsweep] starting recovery N-sweep $(date)"
# Extended recovery N (plain fsync, 3 trials), tagged so it does not clobber the
# primary d=128 CSVs. Also emits the tax/gc at N=2000 (ignored; we read recovery).
make bench-durability DURABILITY_ARGS="--recn 1000,10000,30000,100000 --n 2000 --trials 3 --tag nsweep" \
     > build/durability_nsweep.log 2>&1
echo "[nsweep] COMPLETE $(date)"
touch build/nsweep.done
