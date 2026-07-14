# Linux/NVMe durability experiments (Lever 2 — the generalizability + power-loss story)

These close the paper's biggest external-validity gap: the macOS `~155×` fsync-floor
gap is a platform artifact (macOS plain `fsync` is a page-cache no-op), and the
fork+`_exit` crash test does not exercise real power loss. Both are fixed on a Linux
box; the harnesses are written and committed, you just run them.

## Why this matters for the score
Reviewers unanimously cited "single macOS host; headline number is a platform
artifact" as *the* cap. One Linux/NVMe data point + real power-loss injection moves:
- the durability-tax claim from "plausible elsewhere" → "demonstrated on the platform
  vector DBs actually run on", and
- the crash-consistency claim from "post-`fsync` survival" → "survives power loss,
  torn writes, and reordering."

## Provision (cheap)
Any Linux box with an NVMe SSD, or a cloud spot VM (~\$0.50/hr for an afternoon) with
a couple of scratch block devices/partitions. Then `make bench-durability verify-open`.

## Experiment 1 — honest durability tax on Linux (`scripts/linux_durability.sh`)
On Linux, `fsync(2)` forces a real drive-cache flush, so the **plain-fsync** number
is already honest (no `F_FULLFSYNC` needed).
```
make bench-durability
scripts/linux_durability.sh /mnt/nvme          # existing NVMe mount
# or, no spare device:
scripts/linux_durability.sh --loop 4G ext4     # loopback ext4 (sudo)
```
Report the plain-fsync fsync-floor and tax here next to the macOS `F_FULLFSYNC`
numbers. Expected: the floor gap is far smaller than macOS's 155× (Linux plain fsync
already flushes), which is exactly the point — it quantifies how much of the macOS
number was the platform. Run on ext4 **and** btrfs to show FS sensitivity.

## Experiment 2 — true power-loss crash consistency (`scripts/linux_powerloss.sh`)
Uses `dm-log-writes` to record every block write + FLUSH/FUA barrier, then replays
the log to **each** barrier onto a scratch device and opens the DB there with
`verify_open`. A crash-consistent store opens cleanly at every barrier and never
loses a pre-barrier acknowledged write. This is strictly stronger than fork+`_exit`.
```
# one-time: build xfstests' replay-log
git clone https://git.kernel.org/pub/scm/fs/xfs/xfstests-dev.git
make -C xfstests-dev/src/log-writes

make verify-open
sudo REPLAY_LOG=$PWD/xfstests-dev/src/log-writes/replay-log \
     DATA_DEV=/dev/nvme0n2 LOG_DEV=/dev/nvme0n3 REPLAY_DEV=/dev/nvme0n4 \
     scripts/linux_powerloss.sh
```
Expected: `PASS` — clean open at every barrier, count non-decreasing. Also run a
`--group-commit` variant to confirm the batch's single flush makes the whole batch
atomic under power loss (the C3 crash-consistency claim, now power-loss-grade).

## Folding results into the paper
- Add a Linux column to Table `tab:floor` and Table `tab:tax` (plain fsync = honest);
  state the macOS-vs-Linux floor-gap delta in §3, retiring the "155× is an artifact"
  caveat as *quantified* rather than *asserted*.
- Replace the fork+`_exit` sentence in §5 with the dm-log-writes result ("consistent
  at every FLUSH/FUA barrier under block-level replay").
- Move the corresponding items out of §8 "remaining work".
