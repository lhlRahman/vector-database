#!/usr/bin/env bash
# Destructive Linux dm-log-writes sweep for the recall-aware committer.
#
# Exit 77 means the host/toolchain cannot run this test. Once all capabilities
# and the explicit destructive opt-in are present, workload, mount, recovery, or
# verification errors are test failures (exit 1), never skips.
#
# Example with three disposable block devices:
#
#   sudo VDB_POWERLOSS_ALLOW_DESTRUCTIVE=YES \
#     DATA_DEV=/dev/loop10 LOG_DEV=/dev/loop11 REPLAY_DEV=/dev/loop12 \
#     ARTIFACT_DIR=$PWD/build/powerloss-run scripts/powerloss_committer.sh
#
# The workload/controller writes its ACK ledger under ARTIFACT_DIR, outside the
# logged filesystem. It pauses after persisting each FRONTIER while its parent
# inserts the identically named dm-log-writes mark. This script then replays a
# fresh device prefix at every individual log entry from DB_READY through the
# final crash mark. Each prefix gets normal journal recovery, is remounted
# read-only, and is checked against the latest preceding ACK frontier. fsck -n is
# diagnostic only; this harness never repairs an image.

set -euo pipefail

SKIP_CODE=${SKIP_CODE:-77}

skip() {
    echo "SKIP: $*" >&2
    exit "$SKIP_CODE"
}

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

[[ $(uname -s) == Linux ]] || skip "dm-log-writes requires Linux"
[[ ${EUID:-$(id -u)} -eq 0 ]] || skip "root is required for device-mapper and mounts"
[[ ${VDB_POWERLOSS_ALLOW_DESTRUCTIVE:-} == YES ]] || \
    skip "set VDB_POWERLOSS_ALLOW_DESTRUCTIVE=YES for disposable devices"

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

: "${DATA_DEV:=}"
: "${LOG_DEV:=}"
: "${REPLAY_DEV:=}"
[[ -n $DATA_DEV && -n $LOG_DEV && -n $REPLAY_DEV ]] || \
    skip "set DATA_DEV, LOG_DEV, and REPLAY_DEV to disposable block devices"

REPLAY_LOG=${REPLAY_LOG:-replay-log}
WORKLOAD=${WORKLOAD:-$(pwd)/build/committer_crash_test}
VERIFY=${VERIFY:-$(pwd)/build/verify_committer_image}
FS=${FS:-ext4}
DM_NAME=${DM_NAME:-vdb-committer-$$}
SOURCE_MNT=${SOURCE_MNT:-/mnt/vdb-committer-src-$$}
REPLAY_MNT=${REPLAY_MNT:-/mnt/vdb-committer-replay-$$}
ARTIFACT_DIR=${ARTIFACT_DIR:-$(pwd)/build/powerloss-committer-$(date -u +%Y%m%dT%H%M%SZ)}
LEDGER=$ARTIFACT_DIR/ack-ledger.v1
CUTS=$ARTIFACT_DIR/cuts.tsv
DB_REL=${DB_REL:-db}

for command in dmsetup blockdev mount umount findmnt lsblk awk sed grep sort dd sync sha256sum "$REPLAY_LOG"; do
    command -v "$command" >/dev/null 2>&1 || skip "missing command: $command"
done
command -v "mkfs.$FS" >/dev/null 2>&1 || skip "missing mkfs.$FS"
command -v "fsck.$FS" >/dev/null 2>&1 || skip "missing fsck.$FS"
case "$FS" in
    ext2|ext3|ext4|xfs|btrfs) ;;
    *) [[ -n ${MKFS_ARGS:-} ]] || skip "set MKFS_ARGS for unsupported filesystem $FS" ;;
esac
[[ -x $WORKLOAD ]] || skip "missing committer workload executable: $WORKLOAD"
[[ -x $VERIFY ]] || skip "missing read-only verifier executable: $VERIFY"

set +e
"$VERIFY" --capabilities >/dev/null 2>&1
capability_status=$?
set -e
if [[ $capability_status -eq $SKIP_CODE ]]; then
    skip "committer read-only API is not integrated"
elif [[ $capability_status -ne 0 ]]; then
    fail "verifier capability probe failed with status $capability_status"
fi

for device in "$DATA_DEV" "$LOG_DEV" "$REPLAY_DEV"; do
    [[ -b $device ]] || skip "$device is not a block device"
done

canonical_data=$(readlink -f "$DATA_DEV")
canonical_log=$(readlink -f "$LOG_DEV")
canonical_replay=$(readlink -f "$REPLAY_DEV")
[[ $canonical_data != "$canonical_log" && $canonical_data != "$canonical_replay" &&
   $canonical_log != "$canonical_replay" ]] || fail "DATA_DEV, LOG_DEV, and REPLAY_DEV must be distinct"

device_is_mounted() {
    lsblk -nrpo MOUNTPOINTS "$1" 2>/dev/null | grep -q '[^[:space:]]'
}

for device in "$DATA_DEV" "$LOG_DEV" "$REPLAY_DEV"; do
    device_is_mounted "$device" && fail "$device or one of its children is mounted"
done

data_sectors=$(blockdev --getsz "$DATA_DEV")
replay_sectors=$(blockdev --getsz "$REPLAY_DEV")
[[ $data_sectors -eq $replay_sectors ]] || fail "DATA_DEV and REPLAY_DEV must have identical sector counts"
[[ $data_sectors -gt 0 ]] || fail "DATA_DEV is empty"

replay_help=$({ "$REPLAY_LOG" --help 2>&1 || true; })
for option in --end-mark --find --limit; do
    grep -q -- "$option" <<<"$replay_help" || skip "$REPLAY_LOG lacks required option $option"
done

if ! dmsetup targets 2>/dev/null | grep -q '^log-writes'; then
    if command -v modprobe >/dev/null 2>&1; then
        modprobe dm-log-writes >/dev/null 2>&1 || true
    fi
fi
dmsetup targets 2>/dev/null | grep -q '^log-writes' || skip "kernel dm-log-writes target is unavailable"
dmsetup info "$DM_NAME" >/dev/null 2>&1 && fail "device-mapper name already exists: $DM_NAME"

mkdir -p "$ARTIFACT_DIR" "$SOURCE_MNT" "$REPLAY_MNT"
findmnt -rn -M "$SOURCE_MNT" >/dev/null 2>&1 && fail "$SOURCE_MNT is already a mountpoint"
findmnt -rn -M "$REPLAY_MNT" >/dev/null 2>&1 && fail "$REPLAY_MNT is already a mountpoint"
artifact_real=$(readlink -m "$ARTIFACT_DIR")
source_real=$(readlink -m "$SOURCE_MNT")
replay_real=$(readlink -m "$REPLAY_MNT")
[[ $artifact_real != "$source_real" && $artifact_real != "$source_real"/* &&
   $artifact_real != "$replay_real" && $artifact_real != "$replay_real"/* ]] || \
    fail "ARTIFACT_DIR must be outside the source and replay mountpoints"

source_mounted=0
replay_mounted=0
dm_created=0

cleanup() {
    set +e
    if [[ $replay_mounted -eq 1 ]]; then umount "$REPLAY_MNT" >/dev/null 2>&1; fi
    if [[ $source_mounted -eq 1 ]]; then umount "$SOURCE_MNT" >/dev/null 2>&1; fi
    if [[ $dm_created -eq 1 ]]; then dmsetup remove "$DM_NAME" >/dev/null 2>&1; fi
}
trap cleanup EXIT INT TERM

zero_device() {
    local device=$1
    local sectors full tail
    # blkdiscard is fast for loop/thin devices. The exact-size dd fallback is
    # slower but prevents stale blocks from a later cut leaking into an earlier
    # replay image.
    if command -v blkdiscard >/dev/null 2>&1 && blkdiscard -f "$device" >/dev/null 2>&1; then
        return
    fi
    sectors=$(blockdev --getsz "$device")
    full=$((sectors / 32768))       # 32768 sectors = 16 MiB
    tail=$((sectors % 32768))
    if [[ $full -gt 0 ]]; then
        dd if=/dev/zero of="$device" bs=16777216 count="$full" conv=notrunc,fsync status=none
    fi
    if [[ $tail -gt 0 ]]; then
        dd if=/dev/zero of="$device" bs=512 count="$tail" seek=$((full * 32768)) \
            conv=notrunc,fsync status=none
    fi
}

mkfs_logged_device() {
    case "$FS" in
        ext2|ext3|ext4) "mkfs.$FS" -q -F "/dev/mapper/$DM_NAME" ;;
        xfs|btrfs) "mkfs.$FS" -f "/dev/mapper/$DM_NAME" ;;
        *)
            # Intentional word splitting: MKFS_ARGS is an explicit operator input.
            # shellcheck disable=SC2086
            "mkfs.$FS" $MKFS_ARGS "/dev/mapper/$DM_NAME"
            ;;
    esac
}

echo "[1/5] initialize dm-log-writes and run handshake workload"
zero_device "$DATA_DEV"
zero_device "$LOG_DEV"
zero_device "$REPLAY_DEV"
dmsetup create "$DM_NAME" --table "0 $data_sectors log-writes $DATA_DEV $LOG_DEV"
dm_created=1
mkfs_logged_device
mount "/dev/mapper/$DM_NAME" "$SOURCE_MNT"
source_mounted=1

{
    uname -a
    echo "filesystem=$FS"
    echo "data=$canonical_data sectors=$data_sectors"
    echo "log=$canonical_log sectors=$(blockdev --getsz "$LOG_DEV")"
    echo "replay=$canonical_replay sectors=$replay_sectors"
    echo "workload=$WORKLOAD"
    echo "verifier=$VERIFY"
    dmsetup version
    mount --version || true
    "mkfs.$FS" -V || true
    "fsck.$FS" -V || true
    for device in "$DATA_DEV" "$LOG_DEV" "$REPLAY_DEV"; do
        echo "device=$device logical=$(blockdev --getss "$device") physical=$(blockdev --getpbsz "$device") read_ahead=$(blockdev --getra "$device")"
    done
    lsblk -o NAME,MAJ:MIN,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$DATA_DEV" "$LOG_DEV" "$REPLAY_DEV"
    lsblk -D -o NAME,DISC-ALN,DISC-GRAN,DISC-MAX,DISC-ZERO \
        "$DATA_DEV" "$LOG_DEV" "$REPLAY_DEV" || true
} >"$ARTIFACT_DIR/environment.txt" 2>&1
printf '%s\n' "$replay_help" >"$ARTIFACT_DIR/replay-log-help.txt"

"$WORKLOAD" --physical --dm-name "$DM_NAME" --db "$SOURCE_MNT/$DB_REL" \
    --ledger "$LEDGER" --frontier "${FRONTIER:-wcap}" \
    --dimensions "${DIMENSIONS:-8}" --k "${K:-10}" \
    --stable-records "${STABLE_RECORDS:-16}" --seed "${SEED:-100}" \
    --epsilon "${EPSILON:-0.2}" >"$ARTIFACT_DIR/workload.out" \
    2>"$ARTIFACT_DIR/workload.err" || fail "committer workload failed; see workload.err"

"$VERIFY" --validate-ledger "$LEDGER" >"$ARTIFACT_DIR/ledger-validation.txt" \
    2>&1 || fail "external ACK ledger is invalid"
dmsetup status "$DM_NAME" >"$ARTIFACT_DIR/dm-status.txt"
sync
umount "$SOURCE_MNT"
source_mounted=0
dmsetup remove "$DM_NAME"
dm_created=0

if ! "$REPLAY_LOG" --log "$LOG_DEV" --number-entries \
        >"$ARTIFACT_DIR/log-entry-count.txt" \
        2>"$ARTIFACT_DIR/log-entry-count.stderr"; then
    printf '%s\n' 'number-entries: unsupported by this replay-log build' \
        >"$ARTIFACT_DIR/log-entry-count.txt"
fi

echo "[2/5] resolve named ACK frontiers to block-log entries"
mapfile -t marks < <(awk '$1 == "MARK" { print $2 }' "$LEDGER")
[[ ${#marks[@]} -gt 0 ]] || fail "ledger contains no physical MARK records"
[[ ${marks[0]} == DB_READY ]] || fail "first physical mark must be DB_READY"

declare -a mark_entries
: >"$ARTIFACT_DIR/marks.tsv"
previous_entry=-1
for mark in "${marks[@]}"; do
    location=$("$REPLAY_LOG" --log "$LOG_DEV" --find --end-mark "$mark") || \
        fail "cannot find log mark $mark"
    entry=${location%%@*}
    [[ $entry =~ ^[0-9]+$ ]] || fail "invalid replay-log location for $mark: $location"
    (( entry > previous_entry )) || fail "non-monotone or duplicate log mark $mark"
    mark_entries+=("$entry")
    printf '%s\t%s\t%s\n' "$mark" "$entry" "$location" >>"$ARTIFACT_DIR/marks.tsv"
    previous_entry=$entry
done

first_cut=${mark_entries[0]}
last_mark_index=$((${#marks[@]} - 1))
last_cut=${mark_entries[$last_mark_index]}
total_cuts=$((last_cut - first_cut + 1))
max_cuts=${MAX_CUTS:-0}
if [[ $max_cuts -gt 0 && $total_cuts -gt $max_cuts ]]; then
    fail "cut count $total_cuts exceeds MAX_CUTS=$max_cuts (zero means exhaustive)"
fi

printf 'cut_entry\trequired_frontier\tallow_through\tstatus\n' >"$CUTS"

diagnose_fs() {
    local output=$1
    set +e
    "fsck.$FS" -n "$REPLAY_DEV" >"$output" 2>&1
    set -e
}

replay_prefix() {
    local cut=$1
    zero_device "$REPLAY_DEV"
    "$REPLAY_LOG" --log "$LOG_DEV" --replay "$REPLAY_DEV" --limit $((cut + 1))
}

recover_and_verify() {
    local cut=$1 frontier=$2 upper=$3 output=$4
    if ! mount "$REPLAY_DEV" "$REPLAY_MNT" >"$output.mount-rw" 2>&1; then
        diagnose_fs "$output.fsck-n"
        return 1
    fi
    replay_mounted=1
    if ! umount "$REPLAY_MNT" >>"$output.mount-rw" 2>&1; then
        return 1
    fi
    replay_mounted=0
    if ! mount -o ro "$REPLAY_DEV" "$REPLAY_MNT" >"$output.mount-ro" 2>&1; then
        diagnose_fs "$output.fsck-n"
        return 1
    fi
    replay_mounted=1

    verifier_args=(--db "$REPLAY_MNT/$DB_REL" --ledger "$LEDGER" --frontier "$frontier")
    if [[ $upper != "$frontier" ]]; then
        verifier_args+=(--allow-through "$upper")
    fi
    if ! "$VERIFY" "${verifier_args[@]}" >"$output.verify" 2>&1; then
        diagnose_fs "$output.fsck-n"
        return 1
    fi
    if ! umount "$REPLAY_MNT" >"$output.umount" 2>&1; then
        return 1
    fi
    replay_mounted=0
    return 0
}

echo "[3/5] replay and verify every log entry after DB_READY ($total_cuts cuts)"
mark_cursor=0
for ((cut = first_cut; cut <= last_cut; ++cut)); do
    while (( mark_cursor < last_mark_index && mark_entries[mark_cursor + 1] <= cut )); do
        ((mark_cursor += 1))
    done
    required=${marks[$mark_cursor]}
    upper=$required
    if (( mark_cursor < last_mark_index )); then upper=${marks[$((mark_cursor + 1))]}; fi

    cut_dir=$ARTIFACT_DIR/cut-$cut
    mkdir -p "$cut_dir"
    if ! replay_prefix "$cut" >"$cut_dir/replay.out" 2>"$cut_dir/replay.err"; then
        printf '%s\t%s\t%s\tREPLAY_FAIL\n' "$cut" "$required" "$upper" >>"$CUTS"
        fail "replay failed at log entry $cut"
    fi
    if ! recover_and_verify "$cut" "$required" "$upper" "$cut_dir/check"; then
        printf '%s\t%s\t%s\tVERIFY_FAIL\n' "$cut" "$required" "$upper" >>"$CUTS"
        fail "image failed at log entry $cut (required=$required, allowed-through=$upper)"
    fi
    printf '%s\t%s\t%s\tPASS\n' "$cut" "$required" "$upper" >>"$CUTS"
done

echo "[4/5] archive block-log and filesystem diagnostics"
{
    echo "entries=$(cat "$ARTIFACT_DIR/log-entry-count.txt")"
    echo "first_cut=$first_cut"
    echo "last_cut=$last_cut"
    echo "cuts=$total_cuts"
    echo "marks=${#marks[@]}"
    echo "ledger_sha256=$(sha256sum "$LEDGER" | awk '{print $1}')"
} >"$ARTIFACT_DIR/summary.txt"

echo "[5/5] PASS: $total_cuts physical cuts verified"
echo "artifacts: $ARTIFACT_DIR"
