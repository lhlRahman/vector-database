#!/usr/bin/env bash
# Re-run recall-staleness on SIFT1M with the new adversarial (query-hot) curve,
# after the N-sweep finishes (no CPU contention). Then archive + replot.
set -u
cd /Users/habib/Documents/sig/vector-database
echo "[recall-rerun] waiting for N-sweep to finish ... $(date)"
while pgrep -f 'bench_durability --recn' >/dev/null 2>&1; do sleep 20; done
echo "[recall-rerun] building + running $(date)"
make bench-recall-staleness ANN_ARGS="--data datasets/sift" > build/recall_staleness_sift.log 2>&1
cp build/ann_results/recall_staleness.csv docs/paper/data/recall_staleness_sift1m.csv
/usr/bin/python3 scripts/plot_recall_staleness.py >> build/recall_staleness_sift.log 2>&1
echo "[recall-rerun] COMPLETE $(date)"
touch build/recall_rerun.done
