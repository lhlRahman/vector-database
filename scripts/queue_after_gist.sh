#!/bin/bash
# Runs CPU-heavy benchmarks AFTER the GIST1M competitiveness run finishes, so they
# never contend with its build-time / QPS measurement. Blocks until GIST is done.
set -u
cd /Users/habib/Documents/sig/vector-database

echo "[queue] waiting for GIST1M (bench_ann --data datasets/gist) to finish ... $(date)"
while pgrep -f 'bench_ann --data datasets/gist' >/dev/null 2>&1; do sleep 30; done
echo "[queue] GIST1M finished; starting queued work $(date)"

# 1) hnswlib GIST baseline (matches our GIST run's dataset/params)
echo "[queue] hnswlib GIST baseline ..."
make bench-hnswlib HNSWLIB_ARGS="--data datasets/gist" > build/gist1m_hnswlib.log 2>&1
echo "[queue] hnswlib GIST exit=$?"

# 2) durability tax/recovery on REAL SIFT embeddings (plain + honest), 7 trials.
#    Writes durability_{plain,full}_real_d128.csv (tagged, does not clobber synthetic).
echo "[queue] real-SIFT durability (plain) ..."
make bench-durability DURABILITY_ARGS="--data datasets/sift --n 2000 --trials 7" > build/durability_realsift_plain.log 2>&1
echo "[queue] real-SIFT durability (full-fsync) ..."
make bench-durability DURABILITY_ARGS="--data datasets/sift --n 2000 --trials 7 --full-fsync" > build/durability_realsift_full.log 2>&1

echo "[queue] QUEUE COMPLETE $(date)"
touch build/queue_after_gist.done
