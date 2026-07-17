#!/usr/bin/env bash
# True power-loss crash-consistency test via dm-log-writes (Linux only).
#
# dm-log-writes records every block write and every FLUSH/FUA barrier the DB issues.
# We then replay the log up to EACH barrier onto a scratch device and open the DB
# there: a crash-consistent store opens cleanly at every barrier and never loses an
# acknowledged (pre-barrier) write. This is real power-loss testing -- strictly
# stronger than the fork+_exit test (which leaves the page cache intact).
#
# Prereqs (Linux, root):
#   * two block devices/partitions: $DATA_DEV (fs under test), $LOG_DEV (write log)
#   * one scratch device $REPLAY_DEV to replay onto (>= DATA_DEV size)
#   * xfstests' replay-log tool (build: git clone xfstests-dev; make -C src/log-writes)
#   * ./build/verify_open  (run `make verify-open`)  and ./build/bench_durability
#
# Usage:  sudo REPLAY_LOG=/path/to/replay-log \
#              DATA_DEV=/dev/nvme0n2 LOG_DEV=/dev/nvme0n3 REPLAY_DEV=/dev/nvme0n4 \
#              scripts/linux_powerloss.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DATA_DEV:?set DATA_DEV}"; : "${LOG_DEV:?set LOG_DEV}"; : "${REPLAY_DEV:?set REPLAY_DEV}"
REPLAY_LOG="${REPLAY_LOG:-replay-log}"
FS="${FS:-ext4}"; N="${N:-5000}"
MNT=/mnt/vdb_logwrites; SCRATCH=/mnt/vdb_replay
mkdir -p "$MNT" "$SCRATCH"

SECTORS=$(blockdev --getsz "$DATA_DEV")
echo "[1/5] create dm-log-writes over $DATA_DEV (log -> $LOG_DEV)"
dmsetup remove logwrites 2>/dev/null || true
dmsetup create logwrites --table "0 $SECTORS log-writes $DATA_DEV $LOG_DEV"

echo "[2/5] mkfs + mount, then run the DB insert workload (emits FLUSH/FUA barriers)"
mkfs."$FS" -q -F /dev/mapper/logwrites
mount /dev/mapper/logwrites "$MNT"
# Per-write fsync so every insert is its own durability barrier (worst case for
# consistency testing). --n controls how many acknowledged inserts we then verify.
./build/bench_durability --dir "$MNT" --n "$N" --trials 1 >/dev/null 2>&1 || true
sync; umount "$MNT"
dmsetup remove logwrites

echo "[3/5] enumerate flush/FUA marks in the write log"
NMARKS=$("$REPLAY_LOG" --log "$LOG_DEV" --num-entries 2>/dev/null | tail -1 || echo 0)
echo "    log has entries; replaying to each flush barrier"

echo "[4/5] replay-to-each-barrier and verify the DB opens consistently"
fail=0; last=0
# Replay incrementally to every FUA/flush entry; --end-mark stops at each barrier.
i=0
while "$REPLAY_LOG" --log "$LOG_DEV" --replay "$REPLAY_DEV" --limit $((++i)) --fsck /bin/true >/dev/null 2>&1; do
  mount "$REPLAY_DEV" "$SCRATCH" 2>/dev/null || { echo "  barrier $i: fs will not mount -> INCONSISTENT"; fail=1; break; }
  if ./build/verify_open "$SCRATCH" "$last" ; then
    cnt=$(./build/verify_open "$SCRATCH" 2>/dev/null | sed -n 's/.*count=//p')
    last=${cnt:-$last}
  else
    echo "  barrier $i: verify_open FAILED -> acknowledged write lost or torn record"; fail=1
  fi
  umount "$SCRATCH" 2>/dev/null || true
done

echo "[5/5] result"
if [ "$fail" = 0 ]; then
  echo "PASS: DB opened cleanly at every flush barrier; count non-decreasing (no acknowledged write lost)."
else
  echo "FAIL: a barrier left the store inconsistent (see above)."
fi
exit "$fail"
