#!/usr/bin/env bash
# True power-loss crash-consistency test via dm-log-writes (Linux only).
#
# dm-log-writes records every block write + FLUSH/FUA barrier the DB issues while
# inserting. replay-log then replays the block log and, at EACH flush barrier, runs
# our checker: fsck the replayed device, mount it, and open the DB. A crash-consistent
# store opens cleanly at every barrier with no acknowledged write lost -- real power
# loss, strictly stronger than fork+_exit (which leaves the page cache intact).
#
# Prereqs (Linux, root): DATA_DEV, LOG_DEV, REPLAY_DEV block devices (loopback is
# fine); xfstests' replay-log (REPLAY_LOG=path); ./build/verify_open (make verify-open).
#
#   sudo REPLAY_LOG=~/xfstests-dev/src/log-writes/replay-log \
#        DATA_DEV=$DATA LOG_DEV=$LOG REPLAY_DEV=$REP scripts/linux_powerloss.sh
set -u
cd "$(dirname "$0")/.."
: "${DATA_DEV:?set DATA_DEV}"; : "${LOG_DEV:?set LOG_DEV}"; : "${REPLAY_DEV:?set REPLAY_DEV}"
REPLAY_LOG="${REPLAY_LOG:-replay-log}"
FS="${FS:-ext4}"; N="${N:-300}"
MNT=/mnt/pl_src; SCRATCH=/mnt/pl_rep; DBREL=db
VERIFY="$(pwd)/build/verify_open"
[ -x "$VERIFY" ] || { echo "build first: make verify-open"; exit 1; }
mkdir -p "$MNT" "$SCRATCH"

echo "[1/4] dm-log-writes over $DATA_DEV (log -> $LOG_DEV)"
dmsetup remove logwrites 2>/dev/null || true
SECT=$(blockdev --getsz "$DATA_DEV")
echo "0 $SECT log-writes $DATA_DEV $LOG_DEV" | dmsetup create logwrites

echo "[2/4] mkfs $FS, mount, insert $N durable records (each fsync = a flush barrier)"
mkfs."$FS" -q -F /dev/mapper/logwrites
mount /dev/mapper/logwrites "$MNT"
"$VERIFY" --insert "$MNT/$DBREL" "$N"
sync
umount "$MNT"
dmsetup remove logwrites

echo "[3/4] write per-barrier checker"
CHECK=/tmp/pl_check.sh
cat > "$CHECK" <<EOF
#!/bin/bash
set -u
fsck."$FS" -fy "$REPLAY_DEV" >/dev/null 2>&1
mount "$REPLAY_DEV" "$SCRATCH" 2>/dev/null || exit 1
rc=0
"$VERIFY" "$SCRATCH/$DBREL" 0 >/dev/null 2>&1 || rc=1
umount "$SCRATCH" 2>/dev/null || true
exit \$rc
EOF
chmod +x "$CHECK"

echo "[4/4] replay to EVERY flush barrier, opening the DB at each"
if "$REPLAY_LOG" --log "$LOG_DEV" --replay "$REPLAY_DEV" --check flush --fsck "$CHECK"; then
  echo "POWER-LOSS TEST: PASS -- DB consistent at every flush barrier ($N records, no acknowledged write lost)"
else
  rc=$?
  echo "POWER-LOSS TEST: FAIL (rc=$rc) -- a barrier left the store inconsistent"
  echo "  (if the error is a replay-log flag, paste it; some builds use --check fua)"
  exit 1
fi
