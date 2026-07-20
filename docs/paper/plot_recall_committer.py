#!/usr/bin/env python3
"""Regenerate the summary CSV consumed by the paper's PGFPlots figure."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
AGGREGATE = DATA / "recall_committer.csv"
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


def min_median_max(values):
    return min(values), statistics.median(values), max(values)


def main():
    with AGGREGATE.open(newline="") as handle:
        aggregate = list(csv.DictReader(handle))
    with CRASH.open(newline="") as handle:
        crash = list(csv.DictReader(handle))

    aggregate_by_case = defaultdict(list)
    crash_by_image = defaultdict(list)
    for row in aggregate:
        aggregate_by_case[row["case"]].append(row)
    for row in crash:
        crash_by_image[(row["case"], row["repetition"])].append(row)

    fields = [
        "case",
        "repetitions",
        "crash_observations",
        "writes_per_s_min",
        "writes_per_s_median",
        "writes_per_s_max",
        "weak_p50_ms_min",
        "weak_p50_ms_median",
        "weak_p50_ms_max",
        "stable_p50_ms_min",
        "stable_p50_ms_median",
        "stable_p50_ms_max",
        "query_p95_ms_min",
        "query_p95_ms_median",
        "query_p95_ms_max",
        "max_W",
        "timed_syncs_min",
        "timed_syncs_median",
        "timed_syncs_max",
        "total_syncs_min",
        "total_syncs_median",
        "total_syncs_max",
        "mean_delta_positive",
        "image_mean_delta_min",
        "image_mean_delta_median",
        "image_mean_delta_max",
        "max_delta_positive",
        "max_M",
        "max_amplification",
        "stable_losses",
        "weak_survivors",
        "cap_overshoots",
        "alarm",
    ]
    summaries = []
    for case in ORDER:
        rows = aggregate_by_case[case]
        if not rows:
            raise RuntimeError(f"no aggregate rows for {case}")
        image_rows = {
            repetition: crash_by_image[(case, repetition)]
            for repetition in {row["repetition"] for row in rows}
        }
        if any(not rows_for_image for rows_for_image in image_rows.values()):
            raise RuntimeError(f"missing crash rows for {case}")
        crash_rows = [row for rows_for_image in image_rows.values() for row in rows_for_image]
        throughputs = [number(row, "writes_per_s") for row in rows]
        weak_p50 = [number(row, "weak_p50_us") / 1000.0 for row in rows]
        stable_p50 = [number(row, "stable_p50_us") / 1000.0 for row in rows]
        query_p95 = [number(row, "query_p95_us") / 1000.0 for row in rows]
        timed_syncs = [number(row, "timed_sync_successes") for row in rows]
        total_syncs = [number(row, "total_sync_successes") for row in rows]
        image_delta_means = [
            statistics.mean(number(row, "delta_positive") for row in rows_for_image)
            for rows_for_image in image_rows.values()
        ]
        throughput_min, throughput_median, throughput_max = min_median_max(throughputs)
        weak_min, weak_median, weak_max = min_median_max(weak_p50)
        stable_min, stable_median, stable_max = min_median_max(stable_p50)
        query_min, query_median, query_max = min_median_max(query_p95)
        timed_min, timed_median, timed_max = min_median_max(timed_syncs)
        total_min, total_median, total_max = min_median_max(total_syncs)
        delta_min, delta_median, delta_max = min_median_max(image_delta_means)
        summary = {
                "case": case,
                "repetitions": len(rows),
                "crash_observations": len(crash_rows),
                "writes_per_s_min": throughput_min,
                "writes_per_s_median": throughput_median,
                "writes_per_s_max": throughput_max,
                "weak_p50_ms_min": weak_min,
                "weak_p50_ms_median": weak_median,
                "weak_p50_ms_max": weak_max,
                "stable_p50_ms_min": stable_min,
                "stable_p50_ms_median": stable_median,
                "stable_p50_ms_max": stable_max,
                "query_p95_ms_min": query_min,
                "query_p95_ms_median": query_median,
                "query_p95_ms_max": query_max,
                "max_W": max(int(row["max_W"]) for row in rows),
                "timed_syncs_min": timed_min,
                "timed_syncs_median": timed_median,
                "timed_syncs_max": timed_max,
                "total_syncs_min": total_min,
                "total_syncs_median": total_median,
                "total_syncs_max": total_max,
                "mean_delta_positive": statistics.mean(image_delta_means),
                "image_mean_delta_min": delta_min,
                "image_mean_delta_median": delta_median,
                "image_mean_delta_max": delta_max,
                "max_delta_positive": max(number(row, "delta_positive") for row in crash_rows),
                "max_M": max(number(row, "M") for row in crash_rows),
                "max_amplification": max(
                    number(row, "amplification") for row in crash_rows
                ),
                "stable_losses": sum(int(row["stable_losses"]) for row in rows),
                "weak_survivors": sum(int(row["weak_survivors"]) for row in rows),
                "cap_overshoots": sum(int(row["overshoots"]) for row in rows),
                "alarm": max(int(row["alarm"]) for row in rows),
            }
        summaries.append({
            key: format(value, ".12g") if isinstance(value, float) else value
            for key, value in summary.items()
        })

    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    print(f"wrote {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
