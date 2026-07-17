#!/usr/bin/env bash
set -u; cd /Users/habib/Documents/sig/vector-database
echo "[nsweep-light] waiting for recall-staleness rerun ..."
while pgrep -f 'bench_recall_staleness --data datasets/sift' >/dev/null 2>&1; do sleep 20; done
sleep 5
echo "[nsweep-light] running recovery N-sweep (<=20k, 3 trials)"
make bench-durability DURABILITY_ARGS="--recn 1000,3000,6000,10000,20000 --n 2000 --trials 3 --tag nsweep" > build/durability_nsweep.log 2>&1
echo "[nsweep-light] COMPLETE"; touch build/nsweep.done
