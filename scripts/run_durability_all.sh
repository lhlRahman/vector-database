#!/bin/bash
# Serial durability runs (never concurrent — disk contention corrupts fsync timings).
set -e
BIN=./build/bench_durability
LOG=build/durability_results
mkdir -p "$LOG"
for cfg in "--d 128" "--full-fsync --d 128" "--d 64" "--full-fsync --d 64"; do
  echo "############ RUN: $cfg ############"
  $BIN --n 2000 --trials 7 $cfg
  echo
done
echo "ALL DURABILITY RUNS COMPLETE"
