#!/usr/bin/env python3
"""Regenerate the summary CSV consumed by the paper's PGFPlots figure."""

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
AGGREGATE = DATA / "recall_committer_reported_run.csv"
CRASH = DATA / "recall_committer_crash.csv"
SUMMARY = DATA / "recall_committer_summary.csv"

ORDER = [
    "stable-group",
    "strict-random",
    "strict-hot",
    "exchange-random",
    "exchange-hot",
    "exchange-hot-guard",
]
def number(row, field):
    return float(row[field])


def bootstrap_mean_ci(values, seed, samples=10000):
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def main():
    with AGGREGATE.open(newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    with CRASH.open(newline="") as handle:
        crash = list(csv.DictReader(handle))

    aggregate_by_case = defaultdict(list)
    crash_by_case = defaultdict(list)
    for row in aggregate:
        aggregate_by_case[row["case"]].append(row)
    for row in crash:
        crash_by_case[row["case"]].append(row)

    fields = [
        "case",
        "repetitions",
        "crash_observations",
        "writes_per_s_min",
        "writes_per_s_max",
        "write_p50_ms_min",
        "write_p50_ms_max",
        "query_p95_ms_min",
        "query_p95_ms_max",
        "max_W",
        "syncs_min",
        "syncs_max",
        "mean_delta_positive",
        "mean_delta_ci_low",
        "mean_delta_ci_high",
        "max_delta_positive",
        "max_M",
        "max_amplification",
        "stable_losses",
        "weak_survivors",
        "cap_overshoots",
        "alarm",
    ]
    summaries = []
    for index, case in enumerate(ORDER):
        rows = aggregate_by_case[case]
        crash_rows = crash_by_case[case]
        throughputs = [number(row, "writes_per_s") for row in rows]
        p50 = []
        for row in rows:
            weak = number(row, "weak_p50_us")
            stable = number(row, "stable_p50_us")
            p50.append((weak if weak else stable) / 1000.0)
        query_p95 = [number(row, "query_p95_us") / 1000.0 for row in rows]
        deltas = [number(row, "delta_positive") for row in crash_rows]
        ci_low, ci_high = bootstrap_mean_ci(deltas, 100 + index)
        summaries.append(
            {
                "case": case,
                "repetitions": len(rows),
                "crash_observations": len(crash_rows),
                "writes_per_s_min": min(throughputs),
                "writes_per_s_max": max(throughputs),
                "write_p50_ms_min": min(p50),
                "write_p50_ms_max": max(p50),
                "query_p95_ms_min": min(query_p95),
                "query_p95_ms_max": max(query_p95),
                "max_W": max(int(row["max_W"]) for row in rows),
                "syncs_min": min(int(row["sync_successes"]) for row in rows),
                "syncs_max": max(int(row["sync_successes"]) for row in rows),
                "mean_delta_positive": sum(deltas) / len(deltas),
                "mean_delta_ci_low": ci_low,
                "mean_delta_ci_high": ci_high,
                "max_delta_positive": max(deltas),
                "max_M": max(number(row, "M") for row in crash_rows),
                "max_amplification": max(
                    number(row, "amplification") for row in crash_rows
                ),
                "stable_losses": sum(int(row["stable_losses"]) for row in rows),
                "weak_survivors": sum(int(row["weak_survivors"]) for row in rows),
                "cap_overshoots": sum(int(row["overshoots"]) for row in rows),
                "alarm": max(int(row["alarm"]) for row in rows),
            }
        )

    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"wrote {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
