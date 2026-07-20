#!/usr/bin/env python3
"""Validate and summarize the paired strict-versus-stable throughput sweep."""

import argparse
import csv
import io
import math
import statistics
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


AGGREGATE_NAME = "recall_committer_throughput_sweep.csv"
OPERATIONS_NAME = "recall_committer_throughput_sweep_operations.csv"
SUMMARY_NAME = "recall_committer_throughput_sweep_summary.csv"

PAIR_FIELDS = (
    "workload",
    "k_min",
    "comparison_epsilon",
    "writers",
    "group_delay_us",
    "repetition",
    "hnsw_seed",
    "tail_seed",
)
CONFIG_FIELDS = (
    "workload",
    "k_min",
    "comparison_epsilon",
    "writers",
    "writes",
    "queries",
    "group_delay_us",
)
OP_ID_FIELDS = (
    "case",
    "workload",
    "repetition",
    "hnsw_seed",
    "tail_seed",
    "requested_ack",
    "writers",
    "writes",
    "queries",
    "k_min",
    "epsilon",
    "comparison_epsilon",
    "configured_cap",
    "group_delay_us",
)

AGGREGATE_REQUIRED = set(OP_ID_FIELDS) | {
    "write_seconds",
    "writes_per_s",
    "weak_acks",
    "stable_acks",
    "max_weak",
    "binding",
    "latest_queries",
    "stable_queries",
    "timed_sync_attempts",
    "timed_sync_successes",
    "timed_sync_failures",
    "timed_records_synced",
    "policy_fences",
    "cap_overshoots",
    "all_applied",
}
OPERATIONS_REQUIRED = set(OP_ID_FIELDS) | {
    "index",
    "actual_ack",
    "receipt_cap",
    "weak_records",
}

SUMMARY_FIELDS = (
    "workload",
    "k_min",
    "comparison_epsilon",
    "configured_cap",
    "writers",
    "writes",
    "queries",
    "group_delay_us",
    "repetitions",
    "binding_images",
    "stable_writes_per_s_min",
    "stable_writes_per_s_median",
    "stable_writes_per_s_max",
    "strict_writes_per_s_min",
    "strict_writes_per_s_median",
    "strict_writes_per_s_max",
    "paired_ratio_strict_over_stable_min",
    "paired_ratio_strict_over_stable_median",
    "paired_ratio_strict_over_stable_max",
    "strict_faster_timing_repetitions",
    "stable_timed_syncs_median",
    "strict_timed_syncs_median",
    "stable_timed_records_synced_median",
    "strict_timed_records_synced_median",
    "strict_weak_acks_min",
    "strict_weak_acks_median",
    "strict_weak_acks_max",
    "strict_max_weak",
)


class EvidenceError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing both raw throughput-sweep CSV files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate raw data and require the existing summary to match exactly",
    )
    return parser, parser.parse_args()


def require_headers(path, fieldnames, required):
    missing = sorted(required - set(fieldnames or ()))
    if missing:
        raise EvidenceError(f"{path}: missing columns: {', '.join(missing)}")


def load_csv(path, required):
    if not path.is_file():
        raise EvidenceError(f"required evidence file does not exist: {path}")
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            require_headers(path, reader.fieldnames, required)
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise EvidenceError(f"{path}: expected at least one data row")
    return rows


def location(path, row_number, row):
    case = row.get("case", "?")
    repetition = row.get("repetition", "?")
    return f"{path}: row {row_number} (case={case}, repetition={repetition})"


def integer(row, field, where, minimum=0):
    try:
        value = int(row[field])
    except (KeyError, ValueError) as exc:
        raise EvidenceError(f"{where}: {field} is not an integer: {row.get(field)!r}") from exc
    if value < minimum:
        raise EvidenceError(f"{where}: {field} must be >= {minimum}, got {value}")
    return value


def number(row, field, where, positive=False):
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise EvidenceError(f"{where}: {field} is not numeric: {row.get(field)!r}") from exc
    if not math.isfinite(value):
        raise EvidenceError(f"{where}: {field} must be finite, got {row[field]!r}")
    if positive and value <= 0:
        raise EvidenceError(f"{where}: {field} must be positive, got {value}")
    return value


def decimal_value(row, field, where):
    try:
        value = Decimal(row[field])
    except (KeyError, InvalidOperation) as exc:
        raise EvidenceError(f"{where}: {field} is not decimal: {row.get(field)!r}") from exc
    if not value.is_finite():
        raise EvidenceError(f"{where}: {field} must be finite, got {row[field]!r}")
    return value


def require_equal(actual, expected, where, description):
    if actual != expected:
        raise EvidenceError(f"{where}: {description}: expected {expected!r}, got {actual!r}")


def row_identity(row):
    return tuple(row[field] for field in OP_ID_FIELDS)


def pair_identity(row):
    return tuple(row[field] for field in PAIR_FIELDS)


def config_identity(row):
    return tuple(row[field] for field in CONFIG_FIELDS)


def validate_aggregate(path, rows):
    pairs = defaultdict(list)
    identities = {}
    for row_number, row in enumerate(rows, 2):
        where = location(path, row_number, row)
        ack = row["requested_ack"]
        if ack not in {"stable", "weak"}:
            raise EvidenceError(f"{where}: requested_ack must be stable or weak, got {ack!r}")
        workload = row["workload"]
        if workload not in {"random", "hot"}:
            raise EvidenceError(f"{where}: workload must be random or hot, got {workload!r}")

        writes = integer(row, "writes", where, 1)
        queries = integer(row, "queries", where, 1)
        k_min = integer(row, "k_min", where, 1)
        writers = integer(row, "writers", where, 1)
        integer(row, "group_delay_us", where)
        repetition = integer(row, "repetition", where)
        hnsw_seed = integer(row, "hnsw_seed", where)
        tail_seed = integer(row, "tail_seed", where)
        cap = integer(row, "configured_cap", where)
        weak_acks = integer(row, "weak_acks", where)
        stable_acks = integer(row, "stable_acks", where)
        max_weak = integer(row, "max_weak", where)
        binding = integer(row, "binding", where)
        all_applied = integer(row, "all_applied", where)
        overshoots = integer(row, "cap_overshoots", where)
        latest_queries = integer(row, "latest_queries", where)
        stable_queries = integer(row, "stable_queries", where)
        sync_attempts = integer(row, "timed_sync_attempts", where)
        sync_successes = integer(row, "timed_sync_successes", where)
        sync_failures = integer(row, "timed_sync_failures", where)
        integer(row, "timed_records_synced", where)
        integer(row, "policy_fences", where)
        epsilon = decimal_value(row, "epsilon", where)
        comparison_epsilon = decimal_value(row, "comparison_epsilon", where)
        write_seconds = number(row, "write_seconds", where, positive=True)
        writes_per_s = number(row, "writes_per_s", where, positive=True)

        if not Decimal(0) < comparison_epsilon < Decimal(1):
            raise EvidenceError(
                f"{where}: comparison_epsilon must be in (0, 1), got {comparison_epsilon}"
            )
        expected_rate = writes / write_seconds
        if not math.isclose(writes_per_s, expected_rate, rel_tol=1e-12, abs_tol=1e-12):
            raise EvidenceError(
                f"{where}: writes_per_s {writes_per_s} does not match "
                f"writes/write_seconds {expected_rate}"
            )

        require_equal(weak_acks + stable_acks, writes, where, "ACK count")
        require_equal(latest_queries + stable_queries, queries, where, "query count")
        require_equal(latest_queries, queries, where, "latest-query count")
        require_equal(sync_successes + sync_failures, sync_attempts, where, "sync accounting")
        require_equal(sync_failures, 0, where, "timed sync failures")
        require_equal(all_applied, 1, where, "all_applied")
        require_equal(overshoots, 0, where, "cap_overshoots")
        if binding not in {0, 1}:
            raise EvidenceError(f"{where}: binding must be 0 or 1, got {binding}")
        require_equal(
            hnsw_seed,
            100 if repetition % 2 == 0 else 117,
            where,
            "alternating HNSW seed",
        )
        require_equal(
            tail_seed,
            (45001 if workload == "hot" else 9001) + repetition,
            where,
            "tail seed",
        )

        expected_cap = int(comparison_epsilon * k_min)
        if ack == "stable":
            require_equal(epsilon, Decimal(0), where, "stable epsilon")
            require_equal(cap, 0, where, "stable configured_cap")
            require_equal(weak_acks, 0, where, "stable weak_acks")
            require_equal(stable_acks, writes, where, "stable stable_acks")
            require_equal(max_weak, 0, where, "stable max_weak")
            require_equal(binding, 0, where, "stable binding")
        else:
            require_equal(epsilon, comparison_epsilon, where, "strict comparison epsilon")
            require_equal(cap, expected_cap, where, "strict configured_cap")
            if cap < 1 or weak_acks < 1 or binding != 1:
                raise EvidenceError(
                    f"{where}: strict case did not bind (cap={cap}, weak_acks={weak_acks}, "
                    f"binding={binding})"
                )
            if max_weak > cap:
                raise EvidenceError(f"{where}: max_weak {max_weak} exceeds cap {cap}")

        identity = row_identity(row)
        if identity in identities:
            raise EvidenceError(
                f"{where}: duplicate aggregate identity first seen at row {identities[identity]}"
            )
        identities[identity] = row_number
        pairs[pair_identity(row)].append(row)

    paired = []
    for key, candidates in pairs.items():
        where = f"{path}: pair {key!r}"
        if len(candidates) != 2:
            raise EvidenceError(f"{where}: expected 2 rows, found {len(candidates)}")
        by_ack = {row["requested_ack"]: row for row in candidates}
        if set(by_ack) != {"stable", "weak"}:
            raise EvidenceError(f"{where}: expected one stable and one strict row")
        stable = by_ack["stable"]
        strict = by_ack["weak"]
        for field in ("workload", "writers", "writes", "queries", "k_min", "comparison_epsilon",
                      "group_delay_us", "repetition", "hnsw_seed", "tail_seed"):
            require_equal(strict[field], stable[field], where, f"paired {field}")
        paired.append((stable, strict))

    return paired, {row_identity(row): row for row in rows}


def validate_operations(path, rows, aggregate_by_identity):
    operations = defaultdict(list)
    for row_number, row in enumerate(rows, 2):
        where = location(path, row_number, row)
        identity = row_identity(row)
        if identity not in aggregate_by_identity:
            raise EvidenceError(f"{where}: no matching aggregate row")
        index = integer(row, "index", where)
        weak_records = integer(row, "weak_records", where)
        receipt_cap = integer(row, "receipt_cap", where)
        actual_ack = row["actual_ack"]
        if actual_ack not in {"stable", "weak"}:
            raise EvidenceError(f"{where}: actual_ack must be stable or weak, got {actual_ack!r}")
        aggregate = aggregate_by_identity[identity]
        cap = int(aggregate["configured_cap"])
        require_equal(receipt_cap, cap, where, "receipt cap")
        if weak_records > cap:
            raise EvidenceError(f"{where}: weak_records {weak_records} exceeds cap {cap}")
        if aggregate["requested_ack"] == "stable":
            require_equal(actual_ack, "stable", where, "stable operation ACK")
            require_equal(weak_records, 0, where, "stable operation weak_records")
        operations[identity].append((index, actual_ack, weak_records))

    if set(operations) != set(aggregate_by_identity):
        missing = set(aggregate_by_identity) - set(operations)
        raise EvidenceError(f"{path}: missing operations for {len(missing)} aggregate rows")

    expected_total = 0
    for identity, aggregate in aggregate_by_identity.items():
        where = f"{path}: case={aggregate['case']}, repetition={aggregate['repetition']}"
        expected_writes = int(aggregate["writes"])
        expected_total += expected_writes
        image_ops = operations[identity]
        require_equal(len(image_ops), expected_writes, where, "operation row count")
        indices = sorted(index for index, _, _ in image_ops)
        require_equal(indices, list(range(expected_writes)), where, "operation indices")
        weak_count = sum(ack == "weak" for _, ack, _ in image_ops)
        stable_count = sum(ack == "stable" for _, ack, _ in image_ops)
        require_equal(weak_count, int(aggregate["weak_acks"]), where, "weak operation count")
        require_equal(stable_count, int(aggregate["stable_acks"]), where, "stable operation count")
        require_equal(
            max(weak_records for _, _, weak_records in image_ops),
            int(aggregate["max_weak"]),
            where,
            "maximum weak-record count",
        )
    require_equal(len(rows), expected_total, str(path), "total operation row count")


def min_median_max(values):
    return min(values), statistics.median(values), max(values)


def format_value(value):
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def summarize(pairs):
    by_config = defaultdict(list)
    for stable, strict in pairs:
        by_config[config_identity(stable)].append((stable, strict))

    summaries = []
    for config in sorted(
        by_config,
        key=lambda key: (key[0], int(key[1]), Decimal(key[2]), int(key[3]),
                         int(key[6]), int(key[4]), int(key[5])),
    ):
        image_pairs = by_config[config]
        repetitions = [int(stable["repetition"]) for stable, _ in image_pairs]
        if len(repetitions) != len(set(repetitions)):
            raise EvidenceError(f"configuration {config!r}: duplicate repetition")
        expected_repetitions = list(range(len(repetitions)))
        if sorted(repetitions) != expected_repetitions:
            raise EvidenceError(
                f"configuration {config!r}: repetitions are not contiguous: {sorted(repetitions)}"
            )

        stable_rates = [float(stable["writes_per_s"]) for stable, _ in image_pairs]
        strict_rates = [float(strict["writes_per_s"]) for _, strict in image_pairs]
        ratios = [
            float(strict["writes_per_s"]) / float(stable["writes_per_s"])
            for stable, strict in image_pairs
        ]
        stable_syncs = [int(stable["timed_sync_successes"]) for stable, _ in image_pairs]
        strict_syncs = [int(strict["timed_sync_successes"]) for _, strict in image_pairs]
        stable_records = [int(stable["timed_records_synced"]) for stable, _ in image_pairs]
        strict_records = [int(strict["timed_records_synced"]) for _, strict in image_pairs]
        strict_weak = [int(strict["weak_acks"]) for _, strict in image_pairs]
        stable_min, stable_median, stable_max = min_median_max(stable_rates)
        strict_min, strict_median, strict_max = min_median_max(strict_rates)
        ratio_min, ratio_median, ratio_max = min_median_max(ratios)
        weak_min, weak_median, weak_max = min_median_max(strict_weak)
        first_strict = image_pairs[0][1]
        summary = {
            "workload": config[0],
            "k_min": int(config[1]),
            "comparison_epsilon": float(config[2]),
            "configured_cap": int(first_strict["configured_cap"]),
            "writers": int(config[3]),
            "writes": int(config[4]),
            "queries": int(config[5]),
            "group_delay_us": int(config[6]),
            "repetitions": len(image_pairs),
            "binding_images": sum(int(strict["binding"]) for _, strict in image_pairs),
            "stable_writes_per_s_min": stable_min,
            "stable_writes_per_s_median": stable_median,
            "stable_writes_per_s_max": stable_max,
            "strict_writes_per_s_min": strict_min,
            "strict_writes_per_s_median": strict_median,
            "strict_writes_per_s_max": strict_max,
            "paired_ratio_strict_over_stable_min": ratio_min,
            "paired_ratio_strict_over_stable_median": ratio_median,
            "paired_ratio_strict_over_stable_max": ratio_max,
            "strict_faster_timing_repetitions": sum(ratio > 1.0 for ratio in ratios),
            "stable_timed_syncs_median": statistics.median(stable_syncs),
            "strict_timed_syncs_median": statistics.median(strict_syncs),
            "stable_timed_records_synced_median": statistics.median(stable_records),
            "strict_timed_records_synced_median": statistics.median(strict_records),
            "strict_weak_acks_min": weak_min,
            "strict_weak_acks_median": weak_median,
            "strict_weak_acks_max": weak_max,
            "strict_max_weak": max(int(strict["max_weak"]) for _, strict in image_pairs),
        }
        summaries.append({key: format_value(value) for key, value in summary.items()})
    return summaries


def render_summary(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def run(data_dir, check):
    try:
        data_dir = data_dir.expanduser().resolve(strict=True)
    except OSError as exc:
        raise EvidenceError(f"data directory does not exist: {data_dir}") from exc
    if not data_dir.is_dir():
        raise EvidenceError(f"data path is not a directory: {data_dir}")

    aggregate_path = data_dir / AGGREGATE_NAME
    operations_path = data_dir / OPERATIONS_NAME
    summary_path = data_dir / SUMMARY_NAME
    aggregate = load_csv(aggregate_path, AGGREGATE_REQUIRED)
    operations = load_csv(operations_path, OPERATIONS_REQUIRED)
    pairs, aggregate_by_identity = validate_aggregate(aggregate_path, aggregate)
    validate_operations(operations_path, operations, aggregate_by_identity)
    output = render_summary(summarize(pairs))

    if check:
        if not summary_path.is_file():
            raise EvidenceError(f"summary does not exist for --check: {summary_path}")
        if summary_path.read_bytes() != output:
            raise EvidenceError(
                f"summary is stale or non-deterministic: {summary_path}; regenerate without --check"
            )
        action = "validated"
    else:
        summary_path.write_bytes(output)
        action = "wrote"
    print(
        f"{action} {summary_path} "
        f"({len(pairs)} paired images, {len(output.splitlines()) - 1} configurations)"
    )


def main():
    parser, args = parse_args()
    try:
        run(args.data_dir, args.check)
    except EvidenceError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
