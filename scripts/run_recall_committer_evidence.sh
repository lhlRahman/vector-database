#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPETITIONS=30
OUTPUT="$ROOT/.artifacts/recall-committer-evidence"
STAGING_ROOT="${TMPDIR:-/tmp}/vdb-recall-committer-evidence"
RUN_ID=""
EXISTING=""
INSTALL=1
KEEP_STAGING=0
WORK=""
INSTALL_LOCK=""
INCOMING=""
POINTER_TMP=""
SCHEMA_VERSION=""
SOURCE_SNAPSHOT_ACTIVE=0
SOURCE_SNAPSHOT_VERIFIED=0
SOURCE_MANIFEST_BEFORE=""
HAS_THROUGHPUT_SWEEP=1

readonly SWEEP_WRITES=64
readonly SWEEP_QUERIES=64
readonly SWEEP_K_MINS='10,20,50,100'
readonly SWEEP_EPSILONS='0.2'

readonly AGGREGATE_HEADER_V1='case,workload,repetition,writers,writes_per_s,weak_p50_us,weak_p95_us,weak_p99_us,stable_p50_us,stable_p95_us,stable_p99_us,fence_us,query_p50_us,query_p95_us,query_p99_us,mean_D,mean_W,max_W,max_cap,max_risk,timed_sync_successes,total_sync_attempts,total_sync_successes,total_sync_failures,timed_syncs_per_s,total_records_per_sync,policy_fences,age_fences,overshoots,enrichment,alarm,alarm_latency_ms,M_max,L_max,delta_max,amplification_max,stable_losses,weak_survivors,frontier_ok,recovery_ok,strict_ok'
readonly CRASH_HEADER_V1='case,workload,repetition,query,M,L,delta_positive,amplification,pre_recall,post_recall,answer_churn,durable_overlap,lost_ids_subset_weak,durable_fingerprint_equal'
readonly OPERATIONS_HEADER_V1='case,workload,repetition,hnsw_seed,tail_seed,operation,index,ack,latency_us,lsn,visible_lsn,durable_lsn,durable_records,weak_records,cap,risk,snapshot_lsn,exact_recall,tail_evaluations'
readonly AGGREGATE_HEADER_V2='case,workload,repetition,hnsw_seed,tail_seed,writers,writes_per_s,weak_p50_us,weak_p95_us,weak_p99_us,stable_p50_us,stable_p95_us,stable_p99_us,fence_us,query_p50_us,query_p95_us,query_p99_us,mean_D,mean_W,max_W,max_cap,max_risk,timed_sync_successes,total_sync_attempts,total_sync_successes,total_sync_failures,timed_syncs_per_s,total_records_per_sync,policy_fences,age_fences,overshoots,enrichment,alarm,alarm_latency_ms,crash_frontier,crash_visible_lsn,crash_durable_lsn,crash_child_status,exposed_weak_records,surviving_weak_records,lost_weak_records,has_L_lt_M,M_max,L_max,delta_max,amplification_max,stable_losses,weak_survivors,stable_records_unchanged,cohort_records_unchanged,cohort_expectations_ok,post_recovery_suffix_ok,frontier_ok,recovery_ok,strict_ok'
readonly CRASH_HEADER_V2='case,workload,repetition,hnsw_seed,tail_seed,crash_frontier,crash_visible_lsn,crash_durable_lsn,crash_child_status,exposed_weak_records,surviving_weak_records,lost_weak_records,query,M,L,delta_positive,amplification,pre_recall,post_recall,answer_churn,pre_stable_overlap,lost_ids_subset_weak,pre_merge_fingerprint_equal,recovery_fingerprint_equal,stable_records_unchanged,cohort_records_unchanged,cohort_expectations_ok,post_recovery_suffix_ok'
readonly OPERATIONS_HEADER_V2='case,workload,repetition,hnsw_seed,tail_seed,operation,index,ack,latency_us,lsn,visible_lsn,durable_lsn,durable_records,weak_records,cap,risk,snapshot_lsn,exact_recall,tail_evaluations,crash_frontier,expected_recovery,crash_child_status'
readonly SUMMARY_HEADER_V1='case,repetitions,crash_observations,writes_per_s_min,writes_per_s_median,writes_per_s_max,weak_p50_ms_min,weak_p50_ms_median,weak_p50_ms_max,stable_p50_ms_min,stable_p50_ms_median,stable_p50_ms_max,query_p95_ms_min,query_p95_ms_median,query_p95_ms_max,max_W,timed_syncs_min,timed_syncs_median,timed_syncs_max,total_syncs_min,total_syncs_median,total_syncs_max,mean_delta_positive,image_mean_delta_min,image_mean_delta_median,image_mean_delta_max,max_delta_positive,max_M,max_amplification,stable_losses,weak_survivors,cap_overshoots,alarm'
readonly SUMMARY_HEADER_V2="$SUMMARY_HEADER_V1,max_L,L_lt_M_observations,L_lt_M_images,exposed_weak_records,surviving_weak_records,lost_weak_records"
readonly SWEEP_AGGREGATE_HEADER='case,workload,repetition,hnsw_seed,tail_seed,requested_ack,writers,writes,queries,k_min,epsilon,comparison_epsilon,configured_cap,group_delay_us,write_seconds,writes_per_s,weak_acks,stable_acks,max_weak,binding,weak_p50_us,weak_p95_us,weak_p99_us,stable_p50_us,stable_p95_us,stable_p99_us,query_p50_us,query_p95_us,query_p99_us,latest_queries,stable_queries,timed_sync_attempts,timed_sync_successes,timed_sync_failures,timed_records_synced,policy_fences,follower_requests,fence_then_retry,strict_rejections,cap_overshoots,all_applied'
readonly SWEEP_OPERATIONS_HEADER='case,workload,repetition,hnsw_seed,tail_seed,requested_ack,writers,writes,queries,k_min,epsilon,comparison_epsilon,configured_cap,group_delay_us,index,actual_ack,latency_us,lsn,visible_lsn,durable_lsn,durable_records,weak_records,receipt_cap,risk'

readonly -a BASE_PAYLOAD_FILES=(
    recall_committer.csv
    recall_committer_crash.csv
    recall_committer_operations.csv
    recall_committer_summary.csv
    recall_committer_run.txt
    committer_unit_test.txt
    committer_crash_test.txt
    committer_cut_test.txt
    recall_committer_environment.txt
)
readonly -a SWEEP_PAYLOAD_FILES=(
    recall_committer_throughput_sweep.csv
    recall_committer_throughput_sweep_operations.csv
    recall_committer_throughput_sweep_summary.csv
)
BUNDLE_FILES=("${BASE_PAYLOAD_FILES[@]}" "${SWEEP_PAYLOAD_FILES[@]}" recall_committer_sha256.txt)

usage() {
    cat <<'EOF'
Usage: scripts/run_recall_committer_evidence.sh [options]

Run the committer unit, logical-crash, WAL-cut, main benchmark, and paired
strict-versus-stable throughput sweep in a staging directory. Validate every
artifact before installing a versioned run and atomically switching the
canonical pointer. This script never invokes the physical dm-log-writes harness.

Options:
  --repetitions N          Images per case (default: 30)
  --output DIR             Canonical data directory (default: .artifacts/recall-committer-evidence)
  --staging DIR            Staging parent (default: ${TMPDIR:-/tmp}/vdb-recall-committer-evidence)
  --run-id ID              Version directory name (default: UTC timestamp + commit)
  --validate-existing DIR  Stage and validate an existing flat evidence bundle
  --no-install             Validate only and preserve the staged bundle
  --keep-staging           Preserve staging after a successful install
  -h, --help               Show this help

Relative paths are resolved from the repository root. --validate-existing is
intended for archived-bundle audits; it does not execute tests or the benchmark.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

resolve_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$ROOT" "$1" ;;
    esac
}

need_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repetitions)
            need_value "$@"
            REPETITIONS="$2"
            shift 2
            ;;
        --output)
            need_value "$@"
            OUTPUT="$(resolve_path "$2")"
            shift 2
            ;;
        --staging)
            need_value "$@"
            STAGING_ROOT="$(resolve_path "$2")"
            shift 2
            ;;
        --run-id)
            need_value "$@"
            RUN_ID="$2"
            shift 2
            ;;
        --validate-existing)
            need_value "$@"
            EXISTING="$(resolve_path "$2")"
            shift 2
            ;;
        --no-install)
            INSTALL=0
            shift
            ;;
        --keep-staging)
            KEEP_STAGING=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || die "--repetitions must be a positive integer"
[[ -z "$RUN_ID" || "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "--run-id may contain only letters, digits, dot, underscore, and hyphen"
[[ -z "$EXISTING" || -d "$EXISTING" ]] || die "existing bundle is not a directory: $EXISTING"

command -v git >/dev/null 2>&1 || die "git is required"
command -v awk >/dev/null 2>&1 || die "awk is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

canonical_path() {
    python3 - "$1" <<'PY'
import pathlib
import sys

print(pathlib.Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

resolve_existing_directory() {
    python3 - "$1" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).expanduser().resolve(strict=True)
if not path.is_dir():
    raise SystemExit(f"not a directory: {path}")
print(path)
PY
}

OUTPUT="$(canonical_path "$OUTPUT")"
STAGING_ROOT="$(canonical_path "$STAGING_ROOT")"
BUILD_ROOT="$(canonical_path "$ROOT/build")"
if [[ -n "$EXISTING" ]]; then
    EXISTING="$(resolve_existing_directory "$EXISTING")"
else
    case "$STAGING_ROOT/" in
        "$BUILD_ROOT/"*) die "fresh-run staging cannot be inside $BUILD_ROOT" ;;
    esac
    case "$OUTPUT/" in
        "$BUILD_ROOT/"*) die "fresh-run output cannot be inside $BUILD_ROOT" ;;
    esac
fi

SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_COMMIT:0:12}"
fi

mkdir -p "$STAGING_ROOT"
WORK="$(mktemp -d "$STAGING_ROOT/run.XXXXXX")"
BUNDLE="$WORK/bundle"
mkdir -p "$BUNDLE"

cleanup() {
    local status=$?
    if [[ -n "$POINTER_TMP" && ( -e "$POINTER_TMP" || -L "$POINTER_TMP" ) ]]; then
        rm -f "$POINTER_TMP"
    fi
    if [[ -n "$INCOMING" && -d "$INCOMING" ]]; then
        rm -rf "$INCOMING"
    fi
    if [[ -n "$INSTALL_LOCK" && -d "$INSTALL_LOCK" ]]; then
        rmdir "$INSTALL_LOCK" 2>/dev/null || true
    fi
    if [[ $status -eq 0 && $KEEP_STAGING -eq 0 && $INSTALL -eq 1 ]]; then
        rm -rf "$WORK"
    else
        printf 'staging preserved at %s\n' "$WORK" >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

use_base_bundle_layout() {
    HAS_THROUGHPUT_SWEEP=0
    BUNDLE_FILES=("${BASE_PAYLOAD_FILES[@]}" recall_committer_sha256.txt)
}

use_round4_bundle_layout() {
    HAS_THROUGHPUT_SWEEP=1
    BUNDLE_FILES=("${BASE_PAYLOAD_FILES[@]}" "${SWEEP_PAYLOAD_FILES[@]}" recall_committer_sha256.txt)
}

select_bundle_layout_at() {
    local directory=$1
    local manifest="$directory/recall_committer_sha256.txt"
    [[ -f "$manifest" && ! -L "$manifest" ]] ||
        die "regular manifest missing from $directory"
    if grep -q '  recall_committer_throughput_sweep.csv$' "$manifest"; then
        use_round4_bundle_layout
    else
        use_base_bundle_layout
    fi
}

fsync_directory() {
    python3 - "$1" <<'PY'
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1]).resolve(strict=True)
if not path.is_dir():
    raise SystemExit(f"not a directory: {path}")
flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
fd = os.open(path, flags)
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

fsync_bundle_at() {
    local directory=$1
    shift
    python3 - "$directory" "$@" <<'PY'
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
if not root.is_dir():
    raise SystemExit(f"not a bundle directory: {root}")
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
for name in sys.argv[2:]:
    path = root / name
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise SystemExit(f"not a regular bundle file: {path}")
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
fd = os.open(root, directory_flags)
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

assert_header() {
    local file=$1
    local expected=$2
    local actual
    IFS= read -r actual < "$file" || die "cannot read header: $file"
    [[ "$actual" == "$expected" ]] || {
        printf 'expected: %s\nactual:   %s\n' "$expected" "$actual" >&2
        die "schema mismatch in $(basename "$file")"
    }
}

detect_schema() {
    local actual
    IFS= read -r actual < "$BUNDLE/recall_committer.csv" ||
        die "cannot read aggregate schema"
    if [[ "$actual" == "$AGGREGATE_HEADER_V2" ]]; then
        SCHEMA_VERSION=v2
    elif [[ "$actual" == "$AGGREGATE_HEADER_V1" ]]; then
        SCHEMA_VERSION=v1
    else
        printf 'unknown aggregate header: %s\n' "$actual" >&2
        die "unsupported recall-committer evidence schema"
    fi
}

validate_aggregate_v1() {
    local file="$BUNDLE/recall_committer.csv"
    assert_header "$file" "$AGGREGATE_HEADER_V1"
    awk -F, -v reps="$REPETITIONS" '
        BEGIN {
            split("stable-group fixed-count fixed-time strict-random strict-hot exchange-random exchange-hot exchange-hot-guard", names, " ")
            for (i = 1; i <= 8; ++i) allowed[names[i]] = 1
        }
        NR == 1 { next }
        {
            bad = 0
            if (NF != 41) { print "aggregate field count at line " NR > "/dev/stderr"; bad = 1 }
            if (!($1 in allowed)) { print "unknown aggregate case " $1 > "/dev/stderr"; bad = 1 }
            if ($3 !~ /^[0-9]+$/ || $3 < 0 || $3 >= reps) { print "invalid aggregate repetition at line " NR > "/dev/stderr"; bad = 1 }
            key = $1 SUBSEP $3
            if (++seen[key] != 1) { print "duplicate aggregate image " $1 "," $3 > "/dev/stderr"; bad = 1 }
            if ($24 != 0 || $29 != 0 || $37 != 0 || $38 != 0 || $39 != 1 || $40 != 1 || $41 != 1) {
                print "aggregate invariant failure at line " NR > "/dev/stderr"; bad = 1
            }
            if ($36 + 0 > 1e-12) { print "positive aggregate amplification at line " NR > "/dev/stderr"; bad = 1 }
            if ($1 ~ /^strict-/ && $33 + 0 > 0.200000000001) { print "strict risk bound exceeded at line " NR > "/dev/stderr"; bad = 1 }
            if ($1 == "exchange-hot-guard" && $31 != 1) { print "guard alarm missing at line " NR > "/dev/stderr"; bad = 1 }
            if ($1 != "exchange-hot-guard" && $31 != 0) { print "unexpected alarm at line " NR > "/dev/stderr"; bad = 1 }
            if (bad) failed = 1
            ++rows
        }
        END {
            if (rows != 8 * reps) { print "aggregate rows: expected " 8 * reps ", got " rows > "/dev/stderr"; failed = 1 }
            for (i = 1; i <= 8; ++i) for (r = 0; r < reps; ++r) {
                if (!(names[i] SUBSEP r in seen)) { print "missing aggregate image " names[i] "," r > "/dev/stderr"; failed = 1 }
            }
            exit failed ? 1 : 0
        }
    ' "$file" || die "aggregate CSV validation failed"
}

validate_aggregate_v2() {
    local file="$BUNDLE/recall_committer.csv"
    assert_header "$file" "$AGGREGATE_HEADER_V2"
    awk -F, -v reps="$REPETITIONS" '
        BEGIN {
            split("stable-group fixed-count fixed-time strict-random strict-hot exchange-random exchange-hot exchange-hot-guard", names, " ")
            for (i = 1; i <= 8; ++i) { allowed[names[i]] = 1; case_index[names[i]] = i - 1 }
            workload["stable-group"] = workload["fixed-count"] = workload["fixed-time"] = "random"
            workload["strict-random"] = workload["exchange-random"] = "random"
            workload["strict-hot"] = workload["exchange-hot"] = workload["exchange-hot-guard"] = "hot"
        }
        NR == 1 { next }
        {
            bad = 0
            if (NF != 55) { print "aggregate field count at line " NR > "/dev/stderr"; bad = 1 }
            if (!($1 in allowed) || $2 != workload[$1]) { print "invalid aggregate case/workload at line " NR > "/dev/stderr"; bad = 1 }
            if ($3 !~ /^[0-9]+$/ || $3 < 0 || $3 >= reps) { print "invalid aggregate repetition at line " NR > "/dev/stderr"; bad = 1 }
            key = $1 SUBSEP $3
            if (++seen[key] != 1) { print "duplicate aggregate image " $1 "," $3 > "/dev/stderr"; bad = 1 }
            expected_seed = (($3 + case_index[$1]) % 2 == 0) ? 100 : 117
            expected_tail = ($2 == "hot" ? 45001 : 9001) + $3
            if ($4 != expected_seed || $5 != expected_tail) { print "aggregate seed mismatch at line " NR > "/dev/stderr"; bad = 1 }
            if ($26 != 0 || $31 != 0 || $46 + 0 > 1e-12 || $46 + 0 < -1e-12 ||
                $47 != 0 || $48 != 0 || $49 != 1 || $50 != 1 || $51 != 1 ||
                $52 != 1 || $53 != 1 || $54 != 1 || $55 != 1) {
                print "aggregate invariant failure at line " NR > "/dev/stderr"; bad = 1
            }
            if ($1 ~ /^strict-/ && ($22 + 0 > 0.200000000001 || $43 + 0 > 0.200000000001)) {
                print "strict risk bound exceeded at line " NR > "/dev/stderr"; bad = 1
            }
            if ($1 == "exchange-hot-guard" && $33 != 1) { print "guard alarm missing at line " NR > "/dev/stderr"; bad = 1 }
            if ($1 != "exchange-hot-guard" && $33 != 0) { print "unexpected alarm at line " NR > "/dev/stderr"; bad = 1 }
            if ($1 == "strict-hot") {
                if ($35 != "fence-after-sync-before-publish" || $36 <= $37 || $38 != 86 ||
                    $39 != 2 || $40 != 2 || $41 != 0 || $42 != 1 ||
                    $43 + 0 < 0.199999999999 || $44 + 0 > 1e-12 || $44 + 0 < -1e-12) {
                    print "strict-hot partial frontier mismatch at line " NR > "/dev/stderr"; bad = 1
                }
                ++partial_images
                exposed += $39; surviving += $40; lost += $41
            } else {
                if ($35 != "terminal-unfenced-suffix" || $38 != -1 || $40 != 0 || $41 != $39) {
                    print "terminal frontier mismatch at line " NR > "/dev/stderr"; bad = 1
                }
                ++terminal_images
            }
            if (bad) failed = 1
            ++rows
        }
        END {
            if (rows != 8 * reps) { print "aggregate rows: expected " 8 * reps ", got " rows > "/dev/stderr"; failed = 1 }
            if (partial_images != reps || terminal_images != 7 * reps || exposed != 2 * reps || surviving != 2 * reps || lost != 0) {
                print "partial-survival frontier totals mismatch" > "/dev/stderr"; failed = 1
            }
            for (i = 1; i <= 8; ++i) for (r = 0; r < reps; ++r) {
                if (!(names[i] SUBSEP r in seen)) { print "missing aggregate image " names[i] "," r > "/dev/stderr"; failed = 1 }
            }
            exit failed ? 1 : 0
        }
    ' "$file" || die "aggregate CSV v2 validation failed"
}

validate_aggregate() {
    detect_schema
    if [[ "$SCHEMA_VERSION" == v2 ]]; then
        validate_aggregate_v2
    else
        validate_aggregate_v1
    fi
}

validate_crash_v1() {
    local aggregate="$BUNDLE/recall_committer.csv"
    local file="$BUNDLE/recall_committer_crash.csv"
    assert_header "$file" "$CRASH_HEADER_V1"
    awk -F, -v reps="$REPETITIONS" '
        FNR == NR {
            if (FNR > 1) images[$1 SUBSEP $2 SUBSEP $3] = 1
            next
        }
        FNR == 1 { next }
        {
            bad = 0
            if (NF != 14) { print "crash field count at line " FNR > "/dev/stderr"; bad = 1 }
            image = $1 SUBSEP $2 SUBSEP $3
            if (!(image in images)) { print "crash row without aggregate image at line " FNR > "/dev/stderr"; bad = 1 }
            if ($4 !~ /^[0-9]+$/ || $4 < 0 || $4 >= 12) { print "invalid crash query at line " FNR > "/dev/stderr"; bad = 1 }
            query = image SUBSEP $4
            if (++seen[query] != 1) { print "duplicate crash query at line " FNR > "/dev/stderr"; bad = 1 }
            if ($5 + 1e-12 < $6 || $6 + 1e-12 < $7 || $7 < -1e-12) { print "M/L/delta ordering failure at line " FNR > "/dev/stderr"; bad = 1 }
            if ($8 + 0 > 1e-12 || $8 + 0 < -1e-12) { print "nonzero amplification at line " FNR > "/dev/stderr"; bad = 1 }
            if ($13 != 1 || $14 != 1) { print "crash recovery invariant failure at line " FNR > "/dev/stderr"; bad = 1 }
            if ($1 ~ /^strict-/ && $5 + 0 > 0.200000000001) { print "strict crash risk bound exceeded at line " FNR > "/dev/stderr"; bad = 1 }
            if (bad) failed = 1
            ++count[image]
            ++rows
        }
        END {
            if (rows != 8 * reps * 12) { print "crash rows: expected " 8 * reps * 12 ", got " rows > "/dev/stderr"; failed = 1 }
            for (image in images) {
                if (count[image] != 12) { print "crash rows per image: expected 12, got " count[image] > "/dev/stderr"; failed = 1 }
                for (q = 0; q < 12; ++q) if (!(image SUBSEP q in seen)) {
                    print "missing crash query " q > "/dev/stderr"; failed = 1
                }
            }
            exit failed ? 1 : 0
        }
    ' "$aggregate" "$file" || die "crash CSV validation failed"
}

validate_crash_v2() {
    local aggregate="$BUNDLE/recall_committer.csv"
    local file="$BUNDLE/recall_committer_crash.csv"
    assert_header "$file" "$CRASH_HEADER_V2"
    awk -F, -v reps="$REPETITIONS" '
        BEGIN {
            case_index["stable-group"] = 0; case_index["fixed-count"] = 1
            case_index["fixed-time"] = 2; case_index["strict-random"] = 3
            case_index["strict-hot"] = 4; case_index["exchange-random"] = 5
            case_index["exchange-hot"] = 6; case_index["exchange-hot-guard"] = 7
        }
        FNR == NR {
            if (FNR > 1) images[$1 SUBSEP $2 SUBSEP $3] = 1
            next
        }
        FNR == 1 { next }
        {
            bad = 0
            if (NF != 28) { print "crash field count at line " FNR > "/dev/stderr"; bad = 1 }
            image = $1 SUBSEP $2 SUBSEP $3
            if (!(image in images)) { print "crash row without aggregate image at line " FNR > "/dev/stderr"; bad = 1 }
            expected_seed = (($3 + case_index[$1]) % 2 == 0) ? 100 : 117
            expected_tail = ($2 == "hot" ? 45001 : 9001) + $3
            if ($4 != expected_seed || $5 != expected_tail) { print "crash seed mismatch at line " FNR > "/dev/stderr"; bad = 1 }
            if ($13 !~ /^[0-9]+$/ || $13 < 0 || $13 >= 12) { print "invalid crash query at line " FNR > "/dev/stderr"; bad = 1 }
            query = image SUBSEP $13
            if (++seen[query] != 1) { print "duplicate crash query at line " FNR > "/dev/stderr"; bad = 1 }
            if ($14 + 1e-12 < $15 || $15 + 1e-12 < $16 || $16 < -1e-12) { print "M/L/delta ordering failure at line " FNR > "/dev/stderr"; bad = 1 }
            if ($17 + 0 > 1e-12 || $17 + 0 < -1e-12 ||
                $22 != 1 || $23 != 1 || $24 != 1 || $25 != 1 || $26 != 1 ||
                $27 != 1 || $28 != 1) {
                print "crash recovery invariant failure at line " FNR > "/dev/stderr"; bad = 1
            }
            if ($1 ~ /^strict-/ && $14 + 0 > 0.200000000001) { print "strict crash risk bound exceeded at line " FNR > "/dev/stderr"; bad = 1 }
            if ($1 == "strict-hot") {
                if ($6 != "fence-after-sync-before-publish" || $7 <= $8 || $9 != 86 ||
                    $10 != 2 || $11 != 2 || $12 != 0 || $15 + 0 > 1e-12 || $15 + 0 < -1e-12) {
                    print "strict-hot crash frontier mismatch at line " FNR > "/dev/stderr"; bad = 1
                }
                if ($15 + 1e-12 < $14) gap[image] = 1
            } else if ($6 != "terminal-unfenced-suffix" || $9 != -1 || $11 != 0 || $12 != $10) {
                print "terminal crash frontier mismatch at line " FNR > "/dev/stderr"; bad = 1
            }
            if (bad) failed = 1
            ++count[image]
            ++rows
        }
        END {
            if (rows != 8 * reps * 12) { print "crash rows: expected " 8 * reps * 12 ", got " rows > "/dev/stderr"; failed = 1 }
            for (image in images) {
                if (count[image] != 12) { print "crash rows per image: expected 12, got " count[image] > "/dev/stderr"; failed = 1 }
                for (q = 0; q < 12; ++q) if (!(image SUBSEP q in seen)) {
                    print "missing crash query " q > "/dev/stderr"; failed = 1
                }
                split(image, fields, SUBSEP)
                if (fields[1] == "strict-hot" && !(image in gap)) {
                    print "strict-hot image has no genuine L<M observation" > "/dev/stderr"; failed = 1
                }
            }
            exit failed ? 1 : 0
        }
    ' "$aggregate" "$file" || die "crash CSV v2 validation failed"
}

validate_crash() {
    if [[ "$SCHEMA_VERSION" == v2 ]]; then
        validate_crash_v2
    else
        validate_crash_v1
    fi
}

validate_operations_v1() {
    local aggregate="$BUNDLE/recall_committer.csv"
    local file="$BUNDLE/recall_committer_operations.csv"
    assert_header "$file" "$OPERATIONS_HEADER_V1"
    awk -F, -v reps="$REPETITIONS" '
        FNR == NR {
            if (FNR > 1) images[$1 SUBSEP $2 SUBSEP $3] = 1
            next
        }
        FNR == 1 { next }
        {
            bad = 0
            if (NF != 19) { print "operation field count at line " FNR > "/dev/stderr"; bad = 1 }
            image = $1 SUBSEP $2 SUBSEP $3
            if (!(image in images)) { print "operation without aggregate image at line " FNR > "/dev/stderr"; bad = 1 }
            if ($4 != 100) { print "unexpected HNSW seed at line " FNR > "/dev/stderr"; bad = 1 }
            expected_tail = ($2 == "hot" ? 45001 : 9001) + $3
            if ($5 != expected_tail) { print "unexpected tail seed at line " FNR > "/dev/stderr"; bad = 1 }
            op = $6
            idx = $7
            if (op == "write") {
                if (idx !~ /^[0-9]+$/ || idx < 0 || idx >= 25 || ($8 != "weak" && $8 != "stable")) bad = 1
                ++writes[image]
            } else if (op == "query") {
                if (idx !~ /^[0-9]+$/ || idx < 0 || idx >= 32 || $8 != "" || $18 < 0 || $18 > 1) bad = 1
                ++queries[image]
            } else if (op == "fence") {
                if (idx != 0 || $8 != "stable") bad = 1
                ++fences[image]
            } else {
                bad = 1
            }
            operation = image SUBSEP op SUBSEP idx
            if (++seen[operation] != 1) bad = 1
            if (bad) { print "invalid operation row at line " FNR > "/dev/stderr"; failed = 1 }
            ++rows
        }
        END {
            if (rows != 8 * reps * 58) { print "operation rows: expected " 8 * reps * 58 ", got " rows > "/dev/stderr"; failed = 1 }
            for (image in images) {
                if (writes[image] != 25 || queries[image] != 32 || fences[image] != 1) {
                    print "operation mix mismatch for image" > "/dev/stderr"; failed = 1
                }
            }
            exit failed ? 1 : 0
        }
    ' "$aggregate" "$file" || die "operations CSV validation failed"
}

validate_operations_v2() {
    local aggregate="$BUNDLE/recall_committer.csv"
    local file="$BUNDLE/recall_committer_operations.csv"
    assert_header "$file" "$OPERATIONS_HEADER_V2"
    awk -F, -v reps="$REPETITIONS" '
        BEGIN {
            case_index["stable-group"] = 0; case_index["fixed-count"] = 1
            case_index["fixed-time"] = 2; case_index["strict-random"] = 3
            case_index["strict-hot"] = 4; case_index["exchange-random"] = 5
            case_index["exchange-hot"] = 6; case_index["exchange-hot-guard"] = 7
        }
        FNR == NR {
            if (FNR > 1) images[$1 SUBSEP $2 SUBSEP $3] = 1
            next
        }
        FNR == 1 { next }
        {
            bad = 0
            if (NF != 22) { print "operation field count at line " FNR > "/dev/stderr"; bad = 1 }
            image = $1 SUBSEP $2 SUBSEP $3
            if (!(image in images)) { print "operation without aggregate image at line " FNR > "/dev/stderr"; bad = 1 }
            expected_seed = (($3 + case_index[$1]) % 2 == 0) ? 100 : 117
            expected_tail = ($2 == "hot" ? 45001 : 9001) + $3
            if ($4 != expected_seed || $5 != expected_tail) { print "operation seed mismatch at line " FNR > "/dev/stderr"; bad = 1 }
            expected_frontier = ($1 == "strict-hot") ? "fence-after-sync-before-publish" : "terminal-unfenced-suffix"
            expected_status = ($1 == "strict-hot") ? 86 : -1
            if ($20 != expected_frontier || $22 != expected_status) { print "operation frontier/status mismatch at line " FNR > "/dev/stderr"; bad = 1 }
            op = $6
            idx = $7
            ordinal = rows_in_image[image]++
            if (op == "write") {
                if (idx !~ /^[0-9]+$/ || idx < 0 || idx >= 25 || ($8 != "weak" && $8 != "stable") || $21 != "") bad = 1
                if (ordinal != idx) bad = 1
                ++writes[image]
            } else if (op == "query") {
                if (idx !~ /^[0-9]+$/ || idx < 0 || idx >= 32 || $8 != "" || $18 < 0 || $18 > 1 || $21 != "") bad = 1
                if (ordinal != 25 + idx) bad = 1
                ++queries[image]
            } else if (op == "crash-cohort") {
                if ($1 != "strict-hot" || idx !~ /^[0-9]+$/ || idx < 0 || idx >= 2 ||
                    $8 != "weak" || $21 != "survive" || $10 <= $12 || $10 > $11) bad = 1
                if (ordinal != 58 + idx) bad = 1
                ++cohort[image]
            } else if (op == "crash-fence") {
                if ($1 != "strict-hot" || idx != 0 || $8 != "none" ||
                    $11 <= $12 || $14 != 2 || $21 != "child-exit-after-sync") bad = 1
                if (ordinal != 60) bad = 1
                ++crash_fences[image]
            } else if (op == "timed-prefix-fence") {
                if ($1 != "strict-hot" || idx != 0 || $8 != "stable" || $21 != "") bad = 1
                if (ordinal != 57) bad = 1
                ++final_fences[image]
            } else if (op == "post-recovery-suffix") {
                if ($1 != "strict-hot" || idx != 0 || $8 != "stable" ||
                    $21 != "workflow-resumed" || $12 < $10 || $11 < $12) bad = 1
                if (ordinal != 61) bad = 1
                ++suffixes[image]
            } else if (op == "cleanup-fence") {
                if ($1 == "strict-hot" || idx != 0 || $8 != "stable" || $21 != "") bad = 1
                if (ordinal != 57) bad = 1
                ++final_fences[image]
            } else {
                bad = 1
            }
            operation = image SUBSEP op SUBSEP idx
            if (++seen[operation] != 1) bad = 1
            if (bad) { print "invalid operation row at line " FNR > "/dev/stderr"; failed = 1 }
            ++rows
        }
        END {
            if (rows != 468 * reps) { print "operation rows: expected " 468 * reps ", got " rows > "/dev/stderr"; failed = 1 }
            for (image in images) {
                split(image, fields, SUBSEP)
                expected_cohort = fields[1] == "strict-hot" ? 2 : 0
                expected_crash_fences = fields[1] == "strict-hot" ? 1 : 0
                expected_suffixes = fields[1] == "strict-hot" ? 1 : 0
                if (writes[image] != 25 || queries[image] != 32 || final_fences[image] != 1 ||
                    cohort[image] != expected_cohort || crash_fences[image] != expected_crash_fences ||
                    suffixes[image] != expected_suffixes) {
                    print "operation mix mismatch for image" > "/dev/stderr"; failed = 1
                }
            }
            exit failed ? 1 : 0
        }
    ' "$aggregate" "$file" || die "operations CSV v2 validation failed"
}

validate_operations() {
    if [[ "$SCHEMA_VERSION" == v2 ]]; then
        validate_operations_v2
    else
        validate_operations_v1
    fi
}

validate_summary() {
    local file="$BUNDLE/recall_committer_summary.csv"
    local expected_header expected_fields
    if [[ "$SCHEMA_VERSION" == v2 ]]; then
        expected_header="$SUMMARY_HEADER_V2"
        expected_fields=39
    else
        expected_header="$SUMMARY_HEADER_V1"
        expected_fields=33
    fi
    assert_header "$file" "$expected_header"
    awk -F, -v reps="$REPETITIONS" -v expected_fields="$expected_fields" '
        BEGIN {
            split("stable-group strict-random strict-hot exchange-random exchange-hot exchange-hot-guard", names, " ")
            for (i = 1; i <= 6; ++i) allowed[names[i]] = 1
        }
        NR == 1 { next }
        {
            bad = 0
            if (NF != expected_fields || !($1 in allowed) || ++seen[$1] != 1) bad = 1
            if ($2 != reps || $3 != 12 * reps) bad = 1
            if ($29 + 0 > 1e-12 || $29 + 0 < -1e-12 || $30 != 0 || $31 != 0 || $32 != 0) bad = 1
            if ($1 == "exchange-hot-guard" && $33 != 1) bad = 1
            if ($1 != "exchange-hot-guard" && $33 != 0) bad = 1
            if (bad) { print "invalid summary row at line " NR > "/dev/stderr"; failed = 1 }
            ++rows
        }
        END {
            if (rows != 6) { print "summary rows: expected 6, got " rows > "/dev/stderr"; failed = 1 }
            for (i = 1; i <= 6; ++i) if (!(names[i] in seen)) { print "missing summary case " names[i] > "/dev/stderr"; failed = 1 }
            exit failed ? 1 : 0
        }
    ' "$file" || die "summary CSV validation failed"

    [[ "$SCHEMA_VERSION" == v2 ]] || return 0
    awk -F, \
        -v aggregate="$BUNDLE/recall_committer.csv" \
        -v crash="$BUNDLE/recall_committer_crash.csv" \
        -v summary="$file" '
        function abs(value) { return value < 0 ? -value : value }
        FNR == 1 { next }
        FILENAME == aggregate {
            exposed[$1] += $39
            surviving[$1] += $40
            lost[$1] += $41
            next
        }
        FILENAME == crash {
            m = $14
            l = $15
            if (!(($1) in saw_l) || l > max_l[$1]) max_l[$1] = l
            saw_l[$1] = 1
            if (l + 1e-12 < m) {
                ++gap_observations[$1]
                image = $1 SUBSEP $3
                if (!(image in gap_image_seen)) {
                    gap_image_seen[image] = 1
                    ++gap_images[$1]
                }
            }
            next
        }
        FILENAME == summary {
            bad = 0
            if (abs($34 - max_l[$1]) > 1e-9 || $35 != gap_observations[$1] ||
                $36 != gap_images[$1] || $37 != exposed[$1] ||
                $38 != surviving[$1] || $39 != lost[$1]) bad = 1
            if (bad) {
                print "summary semantic totals mismatch for " $1 > "/dev/stderr"
                failed = 1
            }
        }
        END { exit failed ? 1 : 0 }
    ' "$BUNDLE/recall_committer.csv" "$BUNDLE/recall_committer_crash.csv" "$file" ||
        die "summary semantic validation failed"
}

validate_logs() {
    local unit="$BUNDLE/committer_unit_test.txt"
    local crash="$BUNDLE/committer_crash_test.txt"
    local cut="$BUNDLE/committer_cut_test.txt"
    local run="$BUNDLE/recall_committer_run.txt"
    local count marker

    if [[ $HAS_THROUGHPUT_SWEEP -eq 1 ]]; then
        grep -q 'Results: 77/77 passed' "$unit" || die "base unit-test total is not 77/77"
    elif [[ "$SCHEMA_VERSION" == v2 ]]; then
        grep -q 'Results: 76/76 passed' "$unit" || die "base unit-test total is not 76/76"
    else
        grep -q 'Results: 75/75 passed' "$unit" || die "legacy base unit-test total is not 75/75"
    fi
    grep -q 'All committer policy tests passed' "$unit" || die "committer unit-test success marker missing"
    count="$(grep -c '^PASS ' "$unit" || true)"
    [[ "$count" -eq 24 ]] || die "expected 24 committer unit PASS lines, got $count"

    grep -q 'recall-committer process crash matrix 15/15' "$crash" || die "crash-test total is not 15/15"
    count="$(grep -c '^PASS: process crash frontier=' "$crash" || true)"
    [[ "$count" -eq 15 ]] || die "expected 15 crash frontier PASS lines, got $count"

    grep -q 'committer_cut_test: PASS cuts=661' "$cut" || die "WAL-cut total is not 661"
    if [[ "$SCHEMA_VERSION" == v2 ]]; then
        marker="changed_seed_control_tripped=1 terminal_loss_control_tripped=1 graph_seed_count=2 partial_survival_observed=1 post_recovery_suffixes=$REPETITIONS observed_L_lt_M=1 invariants_ok=1"
        grep -Fq "$marker" "$run" ||
            die "v2 benchmark seed/frontier invariant marker missing"
    else
        grep -q 'changed_seed_control_tripped=1 invariants_ok=1' "$run" ||
            die "legacy benchmark invariant marker missing"
    fi
    if [[ $HAS_THROUGHPUT_SWEEP -eq 1 ]]; then
        grep -q 'sweep_invariants_ok=1' "$run" ||
            die "throughput-sweep invariant marker missing"
    fi
    ! grep -q 'invariants_ok=0' "$run" || die "benchmark reported an invariant failure"
}

regenerate_summary_at() {
    local directory=$1
    python3 "$ROOT/docs/paper/plot_recall_committer.py" \
        --data-dir "$directory" >/dev/null
    if [[ $HAS_THROUGHPUT_SWEEP -eq 1 ]]; then
        python3 "$ROOT/docs/paper/plot_strict_tradeoff.py" \
            --data-dir "$directory" >/dev/null
    fi
}

generate_manifest_at() {
    local directory=$1
    local file
    : > "$directory/recall_committer_sha256.txt"
    for file in "${BUNDLE_FILES[@]}"; do
        [[ "$file" == recall_committer_sha256.txt ]] && continue
        printf '%s  %s\n' "$(sha256_file "$directory/$file")" "$file" >> "$directory/recall_committer_sha256.txt"
    done
}

verify_manifest_at() {
    local directory=$1
    local manifest="$directory/recall_committer_sha256.txt"
    local expected_hash file actual_hash expected_file
    local index=0
    local -a manifest_files=("${BASE_PAYLOAD_FILES[@]}")
    if [[ $HAS_THROUGHPUT_SWEEP -eq 1 ]]; then
        manifest_files+=("${SWEEP_PAYLOAD_FILES[@]}")
    fi
    [[ -f "$manifest" && ! -L "$manifest" ]] || die "regular manifest missing from $directory"
    while read -r expected_hash file; do
        [[ -n "$expected_hash" && -n "$file" ]] || die "malformed manifest line in $manifest"
        [[ "$expected_hash" =~ ^[0-9a-f]{64}$ ]] || die "malformed SHA-256 in $manifest"
        [[ "$file" != /* && "$file" != *'..'* ]] || die "unsafe manifest path: $file"
        [[ $index -lt ${#manifest_files[@]} ]] || die "manifest has unexpected extra member: $file"
        expected_file="${manifest_files[$index]}"
        [[ "$file" == "$expected_file" ]] || die "unexpected manifest member/order: $file"
        [[ -f "$directory/$file" && ! -L "$directory/$file" ]] ||
            die "regular manifest member missing: $file"
        actual_hash="$(sha256_file "$directory/$file")"
        [[ "$actual_hash" == "$expected_hash" ]] || die "SHA-256 mismatch: $file"
        index=$((index + 1))
    done < "$manifest"
    [[ $index -eq ${#manifest_files[@]} ]] || die "manifest has $index entries; expected ${#manifest_files[@]}"
}

validate_throughput_sweep() {
    local aggregate="$BUNDLE/recall_committer_throughput_sweep.csv"
    local operations="$BUNDLE/recall_committer_throughput_sweep_operations.csv"
    local summary="$BUNDLE/recall_committer_throughput_sweep_summary.csv"
    local aggregate_rows operation_rows summary_rows

    assert_header "$aggregate" "$SWEEP_AGGREGATE_HEADER"
    assert_header "$operations" "$SWEEP_OPERATIONS_HEADER"
    python3 "$ROOT/docs/paper/plot_strict_tradeoff.py" \
        --data-dir "$BUNDLE" --check >/dev/null ||
        die "throughput-sweep validation failed"

    aggregate_rows="$(awk 'END { print NR - 1 }' "$aggregate")"
    operation_rows="$(awk 'END { print NR - 1 }' "$operations")"
    summary_rows="$(awk 'END { print NR - 1 }' "$summary")"
    [[ "$aggregate_rows" -eq $((16 * REPETITIONS)) ]] ||
        die "throughput-sweep image rows: expected $((16 * REPETITIONS)), got $aggregate_rows"
    [[ "$operation_rows" -eq $((16 * REPETITIONS * SWEEP_WRITES)) ]] ||
        die "throughput-sweep operation rows: expected $((16 * REPETITIONS * SWEEP_WRITES)), got $operation_rows"
    [[ "$summary_rows" -eq 8 ]] ||
        die "throughput-sweep summary rows: expected 8, got $summary_rows"

    awk -F, -v reps="$REPETITIONS" -v writes="$SWEEP_WRITES" -v queries="$SWEEP_QUERIES" '
        NR == 1 { next }
        {
            bad = 0
            if ($2 != "random" && $2 != "hot") bad = 1
            if ($3 !~ /^[0-9]+$/ || $3 < 0 || $3 >= reps) bad = 1
            if ($7 != 4 || $8 != writes || $9 != queries) bad = 1
            if ($10 != 10 && $10 != 20 && $10 != 50 && $10 != 100) bad = 1
            if ($12 + 0 < 0.199999999999 || $12 + 0 > 0.200000000001) bad = 1
            if ($14 != 750) bad = 1
            if (bad) {
                print "unexpected throughput-sweep configuration at line " NR > "/dev/stderr"
                failed = 1
            }
        }
        END { exit failed ? 1 : 0 }
    ' "$aggregate" || die "throughput-sweep fixed-design validation failed"
}

validate_bundle() {
    local file
    for file in "${BUNDLE_FILES[@]}"; do
        [[ -s "$BUNDLE/$file" ]] || die "missing or empty bundle member: $file"
    done
    validate_aggregate
    validate_crash
    validate_operations
    validate_summary
    validate_logs
    if [[ $HAS_THROUGHPUT_SWEEP -eq 1 ]]; then
        validate_throughput_sweep
    fi
    verify_manifest_at "$BUNDLE"
}

relevant_source_status() {
    git -C "$ROOT" status --porcelain --untracked-files=all -- \
        Makefile include src test docs/paper/plot_recall_committer.py \
        docs/paper/plot_strict_tradeoff.py \
        scripts/run_recall_committer_evidence.sh
}

create_source_manifest() {
    local manifest=$1
    local file_list="${manifest}.files"
    local path relative
    git -C "$ROOT" ls-files -- \
        Makefile include src test docs/paper/plot_recall_committer.py \
        docs/paper/plot_strict_tradeoff.py scripts/run_recall_committer_evidence.sh |
        while IFS= read -r relative; do
            printf '%s/%s\n' "$ROOT" "$relative"
        done | LC_ALL=C sort -u > "$file_list"
    : > "$manifest"
    while IFS= read -r path; do
        relative="${path#"$ROOT"/}"
        printf '%s  %s\n' "$(sha256_file "$path")" "$relative" >> "$manifest"
    done < "$file_list"
}

begin_source_snapshot() {
    local status current_commit
    status="$(relevant_source_status)"
    if [[ -n "$status" ]]; then
        printf '%s\n' "$status" >&2
        die "benchmark, test, plot, or runner sources must be clean"
    fi
    current_commit="$(git -C "$ROOT" rev-parse HEAD)"
    [[ "$current_commit" == "$SOURCE_COMMIT" ]] || die "HEAD changed before source snapshot"
    SOURCE_MANIFEST_BEFORE="$WORK/relevant_source_sha256.before.txt"
    create_source_manifest "$SOURCE_MANIFEST_BEFORE"
    [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] ||
        die "HEAD changed while taking source snapshot"
    [[ -z "$(relevant_source_status)" ]] ||
        die "relevant sources changed while taking source snapshot"
    SOURCE_SNAPSHOT_ACTIVE=1
}

assert_source_snapshot_unchanged() {
    local status after_manifest="$WORK/relevant_source_sha256.after.txt"
    [[ $SOURCE_SNAPSHOT_ACTIVE -eq 1 ]] || die "source snapshot was not initialized"
    [[ "$(git -C "$ROOT" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] ||
        die "HEAD changed during evidence execution"
    status="$(relevant_source_status)"
    if [[ -n "$status" ]]; then
        printf '%s\n' "$status" >&2
        die "relevant sources changed during evidence execution"
    fi
    create_source_manifest "$after_manifest"
    cmp -s "$SOURCE_MANIFEST_BEFORE" "$after_manifest" ||
        die "relevant source hashes changed during evidence execution"
    SOURCE_SNAPSHOT_VERIFIED=1
}

capture_environment() {
    local status relevant_status diff_hash source_manifest_hash
    local os compiler cpu memory filesystem aggregate_rows crash_rows operation_rows
    local graph_seed_100_rows graph_seed_117_rows
    local partial_images terminal_images exposed surviving lost child_status_86 suffix_rows
    local sweep_rows sweep_operation_rows
    [[ $SOURCE_SNAPSHOT_VERIFIED -eq 1 ]] || die "source snapshot was not verified"
    status="$(git -C "$ROOT" status --porcelain --untracked-files=all || true)"
    relevant_status="$(relevant_source_status)"
    git -C "$ROOT" diff --binary HEAD -- Makefile include src test \
        docs/paper/plot_recall_committer.py docs/paper/plot_strict_tradeoff.py \
        scripts/run_recall_committer_evidence.sh \
        > "$WORK/relevant_source.diff"
    diff_hash="$(sha256_file "$WORK/relevant_source.diff")"
    source_manifest_hash="$(sha256_file "$SOURCE_MANIFEST_BEFORE")"
    os="$(uname -srvmo 2>/dev/null || uname -a)"
    compiler="$(c++ --version 2>/dev/null | head -n 1 || true)"
    cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || awk -F: '/model name/ {sub(/^ /, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || true)"
    memory="$(sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal/ {print $2 " kB"}' /proc/meminfo 2>/dev/null || true)"
    filesystem="$(df -P "$BUNDLE" 2>/dev/null | tail -n 1 || true)"
    aggregate_rows="$(awk 'END { print NR - 1 }' "$BUNDLE/recall_committer.csv")"
    crash_rows="$(awk 'END { print NR - 1 }' "$BUNDLE/recall_committer_crash.csv")"
    operation_rows="$(awk 'END { print NR - 1 }' "$BUNDLE/recall_committer_operations.csv")"
    sweep_rows="$(awk 'END { print NR - 1 }' "$BUNDLE/recall_committer_throughput_sweep.csv")"
    sweep_operation_rows="$(awk 'END { print NR - 1 }' "$BUNDLE/recall_committer_throughput_sweep_operations.csv")"
    graph_seed_100_rows="$(awk -F, 'NR > 1 && $4 == 100 { ++n } END { print n + 0 }' "$BUNDLE/recall_committer_operations.csv")"
    graph_seed_117_rows="$(awk -F, 'NR > 1 && $4 == 117 { ++n } END { print n + 0 }' "$BUNDLE/recall_committer_operations.csv")"
    suffix_rows="$(awk -F, 'NR > 1 && $6 == "post-recovery-suffix" { ++n } END { print n + 0 }' "$BUNDLE/recall_committer_operations.csv")"
    partial_images=0; terminal_images=0; exposed=0; surviving=0; lost=0; child_status_86=0
    if [[ "$SCHEMA_VERSION" == v2 ]]; then
        read -r partial_images terminal_images exposed surviving lost child_status_86 <<EOF
$(awk -F, 'NR > 1 {
    if ($35 == "fence-after-sync-before-publish") {
        ++partial; exposed += $39; surviving += $40; lost += $41
        if ($38 == 86) ++child86
    } else if ($35 == "terminal-unfenced-suffix") ++terminal
} END { print partial + 0, terminal + 0, exposed + 0, surviving + 0, lost + 0, child86 + 0 }' "$BUNDLE/recall_committer.csv")
EOF
    fi
    {
        printf 'artifact=recall-committer staged evidence run\n'
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'run_date_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'benchmark_source_commit=%s\n' "$SOURCE_COMMIT"
        printf 'benchmark_source_commit_after=%s\n' "$(git -C "$ROOT" rev-parse HEAD)"
        printf 'source_snapshot_verified_after_execution=yes\n'
        printf 'evidence_schema=%s\n' "$SCHEMA_VERSION"
        printf 'relevant_source_diff_sha256=%s\n' "$diff_hash"
        printf 'relevant_source_manifest_sha256=%s\n' "$source_manifest_hash"
        printf 'command=make clean; make committer-unit-test; make committer-crash-test; make committer-cut-test; make build/bench_recall_commit; ./build/bench_recall_commit --repetitions %s --output <staging>; ./build/bench_recall_commit --throughput-sweep --repetitions %s --writes %s --queries %s --sweep-k-mins %s --sweep-epsilons %s --output <staging>\n' \
            "$REPETITIONS" "$REPETITIONS" "$SWEEP_WRITES" "$SWEEP_QUERIES" \
            "$SWEEP_K_MINS" "$SWEEP_EPSILONS"
        printf 'physical_powerloss_harness_invoked=no\n'
        printf 'crash_model=logical live-directory image opened through production read-only recovery\n'
        printf 'repetitions_per_case=%s\n' "$REPETITIONS"
        printf 'cases=8\n'
        printf 'aggregate_image_rows=%s\n' "$aggregate_rows"
        printf 'recovered_query_rows=%s\n' "$crash_rows"
        printf 'raw_operation_rows=%s\n' "$operation_rows"
        printf 'throughput_sweep_image_rows=%s\n' "$sweep_rows"
        printf 'throughput_sweep_operation_rows=%s\n' "$sweep_operation_rows"
        printf 'throughput_sweep_writes_per_image=%s\n' "$SWEEP_WRITES"
        printf 'throughput_sweep_queries_per_image=%s\n' "$SWEEP_QUERIES"
        printf 'throughput_sweep_k_mins=%s\n' "$SWEEP_K_MINS"
        printf 'throughput_sweep_epsilons=%s\n' "$SWEEP_EPSILONS"
        printf 'base_records=160\n'
        printf 'writes_per_image=25\n'
        printf 'concurrent_queries_per_image=32\n'
        printf 'recovery_queries_per_image=12\n'
        printf 'dimensions=12\n'
        printf 'k=10\n'
        printf 'writers=4\n'
        printf 'hnsw_graph_seeds=%s\n' "$([[ "$SCHEMA_VERSION" == v2 ]] && printf '100,117' || printf '100')"
        printf 'graph_seed_100_operation_rows=%s\n' "$graph_seed_100_rows"
        printf 'graph_seed_117_operation_rows=%s\n' "$graph_seed_117_rows"
        printf 'tail_seeds=random:9001+repetition,hot:45001+repetition\n'
        printf 'partial_frontier=fence-after-sync-before-publish\n'
        printf 'terminal_frontier=terminal-unfenced-suffix\n'
        printf 'partial_frontier_images=%s\n' "$partial_images"
        printf 'terminal_frontier_images=%s\n' "$terminal_images"
        printf 'partial_frontier_child_status_86_images=%s\n' "$child_status_86"
        printf 'partial_frontier_exposed_weak_records=%s\n' "$exposed"
        printf 'partial_frontier_surviving_weak_records=%s\n' "$surviving"
        printf 'partial_frontier_lost_weak_records=%s\n' "$lost"
        printf 'post_recovery_suffix_rows=%s\n' "$suffix_rows"
        printf 'statistical_unit=one independently executed tail/write live image\n'
        printf 'uncertainty_summary=min/median/max across images; no query-row bootstrap\n'
        printf 'os=%s\n' "$os"
        printf 'cpu=%s\n' "$cpu"
        printf 'memory=%s\n' "$memory"
        printf 'compiler=%s\n' "$compiler"
        printf 'filesystem_df=%s\n' "$filesystem"
        printf 'relevant_git_status_porcelain_begin\n%s\nrelevant_git_status_porcelain_end\n' "${relevant_status:-<clean>}"
        printf 'git_status_porcelain_begin\n%s\ngit_status_porcelain_end\n' "${status:-<clean>}"
        printf 'relevant_source_sha256_begin\n'
        cat "$SOURCE_MANIFEST_BEFORE"
        printf 'relevant_source_sha256_end\n'
    } > "$BUNDLE/recall_committer_environment.txt"
}

copy_existing_bundle() {
    local file
    select_bundle_layout_at "$EXISTING"
    verify_manifest_at "$EXISTING"
    for file in "${BUNDLE_FILES[@]}"; do
        [[ -s "$EXISTING/$file" ]] || die "existing bundle member missing: $EXISTING/$file"
        cp -p "$EXISTING/$file" "$BUNDLE/$file"
    done
    verify_manifest_at "$BUNDLE"
    verify_manifest_at "$EXISTING"
    for file in "${BUNDLE_FILES[@]}"; do
        cmp -s "$EXISTING/$file" "$BUNDLE/$file" ||
            die "existing bundle changed while copying: $file"
    done
}

reproduce_existing_derived_artifacts() {
    local reproduced="$WORK/reproduced"
    local file
    mkdir -p "$reproduced"
    for file in "${BUNDLE_FILES[@]}"; do
        case "$file" in
            recall_committer_summary.csv|recall_committer_throughput_sweep_summary.csv|recall_committer_sha256.txt) continue ;;
        esac
        cp -p "$BUNDLE/$file" "$reproduced/$file"
    done
    regenerate_summary_at "$reproduced"
    generate_manifest_at "$reproduced"
    cmp -s "$BUNDLE/recall_committer_summary.csv" \
        "$reproduced/recall_committer_summary.csv" ||
        die "existing summary is not byte-for-byte reproducible"
    if [[ $HAS_THROUGHPUT_SWEEP -eq 1 ]]; then
        cmp -s "$BUNDLE/recall_committer_throughput_sweep_summary.csv" \
            "$reproduced/recall_committer_throughput_sweep_summary.csv" ||
            die "existing throughput-sweep summary is not byte-for-byte reproducible"
    fi
    cmp -s "$BUNDLE/recall_committer_sha256.txt" \
        "$reproduced/recall_committer_sha256.txt" ||
        die "existing SHA-256 manifest is not byte-for-byte reproducible"
    verify_manifest_at "$reproduced"
}

assert_existing_bundle_unchanged() {
    local file
    verify_manifest_at "$EXISTING"
    for file in "${BUNDLE_FILES[@]}"; do
        cmp -s "$EXISTING/$file" "$BUNDLE/$file" ||
            die "existing bundle changed during validation: $file"
    done
}

run_evidence() {
    use_round4_bundle_layout
    begin_source_snapshot
    (
        cd "$ROOT"
        make clean
        make committer-unit-test
    ) 2>&1 | tee "$BUNDLE/committer_unit_test.txt"
    (
        cd "$ROOT"
        make committer-crash-test
    ) 2>&1 | tee "$BUNDLE/committer_crash_test.txt"
    (
        cd "$ROOT"
        make committer-cut-test
    ) 2>&1 | tee "$BUNDLE/committer_cut_test.txt"
    (
        cd "$ROOT"
        make build/bench_recall_commit
        ./build/bench_recall_commit --repetitions "$REPETITIONS" --output "$BUNDLE"
    ) 2>&1 | tee "$BUNDLE/recall_committer_run.txt"
    (
        cd "$ROOT"
        ./build/bench_recall_commit \
            --throughput-sweep \
            --repetitions "$REPETITIONS" \
            --writes "$SWEEP_WRITES" \
            --queries "$SWEEP_QUERIES" \
            --sweep-k-mins "$SWEEP_K_MINS" \
            --sweep-epsilons "$SWEEP_EPSILONS" \
            --output "$BUNDLE"
    ) 2>&1 | tee -a "$BUNDLE/recall_committer_run.txt"
    detect_schema
}

install_bundle() {
    local runs final pointer file output_parent
    output_parent="$(dirname "$OUTPUT")"
    mkdir -p "$OUTPUT"
    fsync_directory "$output_parent"
    fsync_directory "$OUTPUT"
    runs="$OUTPUT/recall_committer_runs"
    mkdir -p "$runs"
    fsync_directory "$OUTPUT"
    fsync_directory "$runs"
    INSTALL_LOCK="$OUTPUT/.recall_committer_install.lock"
    mkdir "$INSTALL_LOCK" 2>/dev/null || die "another evidence install is active: $INSTALL_LOCK"

    final="$runs/$RUN_ID"
    [[ ! -e "$final" ]] || die "versioned run already exists: $final"
    pointer="$OUTPUT/recall_committer_canonical"
    [[ ! -d "$pointer" || -L "$pointer" ]] || die "canonical pointer path is a real directory: $pointer"
    POINTER_TMP="$OUTPUT/.recall_committer_canonical.${RUN_ID}.$$"
    [[ ! -e "$POINTER_TMP" && ! -L "$POINTER_TMP" ]] || die "temporary pointer path exists: $POINTER_TMP"

    INCOMING="$(mktemp -d "$runs/.incoming.${RUN_ID}.XXXXXX")"
    for file in "${BUNDLE_FILES[@]}"; do
        cp -p "$BUNDLE/$file" "$INCOMING/$file"
    done
    verify_manifest_at "$INCOMING"
    fsync_bundle_at "$INCOMING" "${BUNDLE_FILES[@]}"
    fsync_directory "$runs"
    mv "$INCOMING" "$final"
    INCOMING=""
    fsync_directory "$runs"

    ln -s "recall_committer_runs/$RUN_ID" "$POINTER_TMP"
    fsync_directory "$OUTPUT"
    python3 - "$POINTER_TMP" "$pointer" <<'PY'
import os
import sys

os.replace(sys.argv[1], sys.argv[2])
PY
    POINTER_TMP=""
    fsync_directory "$OUTPUT"
    rmdir "$INSTALL_LOCK"
    INSTALL_LOCK=""
    fsync_directory "$OUTPUT"
    printf 'installed versioned bundle: %s\n' "$final"
    printf 'canonical bundle: %s -> recall_committer_runs/%s\n' "$pointer" "$RUN_ID"
}

if [[ -n "$EXISTING" ]]; then
    printf 'staging existing evidence from %s\n' "$EXISTING"
    copy_existing_bundle
    validate_bundle
    reproduce_existing_derived_artifacts
    assert_existing_bundle_unchanged
else
    run_evidence
    validate_aggregate
    validate_crash
    validate_operations
    validate_logs
    regenerate_summary_at "$BUNDLE"
    assert_source_snapshot_unchanged
    capture_environment
    generate_manifest_at "$BUNDLE"
    validate_bundle
    assert_source_snapshot_unchanged
fi

fsync_bundle_at "$BUNDLE" "${BUNDLE_FILES[@]}"
fsync_directory "$WORK"
fsync_directory "$STAGING_ROOT"
if [[ "$SCHEMA_VERSION" == v2 ]]; then
    EXPECTED_OPERATION_ROWS=$((468 * REPETITIONS))
else
    EXPECTED_OPERATION_ROWS=$((8 * REPETITIONS * 58))
fi
printf 'validated %d aggregate, %d crash, and %d operation rows\n' \
    "$((8 * REPETITIONS))" "$((8 * REPETITIONS * 12))" "$EXPECTED_OPERATION_ROWS"
if [[ $HAS_THROUGHPUT_SWEEP -eq 1 ]]; then
    printf 'validated %d paired-sweep images and %d paired-sweep operation rows\n' \
        "$((16 * REPETITIONS))" "$((16 * REPETITIONS * SWEEP_WRITES))"
fi

if [[ $INSTALL -eq 1 ]]; then
    if [[ $SOURCE_SNAPSHOT_ACTIVE -eq 1 ]]; then
        assert_source_snapshot_unchanged
    fi
    install_bundle
else
    KEEP_STAGING=1
    printf 'validated staged bundle: %s\n' "$BUNDLE"
fi
