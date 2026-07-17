#!/usr/bin/env bash
# Honest durability tax on Linux/NVMe — the experiment macOS can't do.
#
# On Linux, fsync(2) DOES force a drive-cache flush on a correctly-configured
# device, so the plain-fsync number here is already honest (unlike macOS, where
# plain fsync is a page-cache no-op and F_FULLFSYNC is required). This script runs
# the durability benchmark on a real filesystem/device and records the tax where it
# generalizes.
#
# Usage:
#   scripts/linux_durability.sh /mnt/nvme            # run on an existing NVMe mount
#   scripts/linux_durability.sh --loop 4G ext4       # create a 4G loopback ext4 and run
#
# Prereqs: a Linux host, a built ./build/bench_durability (run `make bench-durability`
# once to build), and write access to the target dir (or sudo for --loop).
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/bench_durability
[ -x "$BIN" ] || { echo "build first:  make bench-durability"; exit 1; }

if [ "${1:-}" = "--loop" ]; then
  SIZE="${2:-4G}"; FS="${3:-ext4}"
  IMG=$(mktemp /tmp/vdb_loop.XXXX.img); truncate -s "$SIZE" "$IMG"
  MNT=$(mktemp -d /tmp/vdb_mnt.XXXX)
  echo "creating $FS on loopback $IMG -> $MNT (needs sudo)"
  sudo mkfs."$FS" -q "$IMG"
  sudo mount -o loop "$IMG" "$MNT"; sudo chown "$(id -u):$(id -g)" "$MNT"
  DIR="$MNT"
  cleanup() { sudo umount "$MNT" 2>/dev/null || true; rm -f "$IMG"; rmdir "$MNT" 2>/dev/null || true; }
  trap cleanup EXIT
else
  DIR="${1:?usage: linux_durability.sh <dir> | --loop <size> <fs>}"
fi

echo "== filesystem =="; df -T "$DIR" | tail -1
echo "== device write-cache (want 'write back' for a realistic flush cost) =="
DEV=$(df "$DIR" | tail -1 | awk '{print $1}')
lsblk -o NAME,ROTA,DISC-GRAN,MODEL "$DEV" 2>/dev/null || true

# On Linux plain fsync is honest, so we report it directly (no F_FULLFSYNC needed).
# We still run both label modes for parity with the macOS tables.
for mode in "" "--full-fsync"; do
  echo "############ Linux durability ${mode:-plain} on $DIR ############"
  "$BIN" --dir "$DIR" --n 2000 --trials 7 --d 128 $mode
done
echo "Done. Compare the plain-fsync tax here (honest on Linux) to the macOS/APFS"
echo "F_FULLFSYNC numbers in the paper; the floor gap should be far smaller than 155x."
