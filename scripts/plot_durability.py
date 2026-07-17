#!/usr/bin/env python3
"""Plot the durability results (median with min-max ranges over K trials).

Reads build/durability_results/durability_{plain,full}_d{128,64}.csv (schema:
metric,d,mode,median,min,max) and writes figures to docs/paper/figs/:
  * durability_tax.pdf       throughput + p99, plain vs F_FULLFSYNC (d=128)
  * groupcommit_sweep.pdf    throughput vs batch size (the amortization curve)
  * recovery_sweep.pdf       sealed vs mutable recovery time vs N

    /usr/bin/python3 scripts/plot_durability.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "build", "durability_results")
OUT = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

PLAIN_C, FULL_C = "#1f77b4", "#d62728"


def load(mode, d):
    path = os.path.join(RES, f"durability_{mode}_d{d}.csv")
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["metric"]] = (float(r["median"]), float(r["min"]), float(r["max"]))
    return m


def err(stat):
    """(median, [[low_err],[high_err]]) for asymmetric matplotlib error bars."""
    med, lo, hi = stat
    return med, [[med - lo], [hi - med]]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


plain, full = load("plain", 128), load("full", 128)
if plain is None or full is None:
    raise SystemExit("d=128 CSVs not found yet — run scripts/run_durability_all.sh first")

# --- Fig A: durability tax (throughput + p99), plain vs F_FULLFSYNC ------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.4))
modes = ["Plain fsync", "F_FULLFSYNC"]
tp = [err(plain["perwrite_qps"]), err(full["perwrite_qps"])]
a1.bar(modes, [t[0] for t in tp], yerr=[[t[1][0][0] for t in tp], [t[1][1][0] for t in tp]],
       color=[PLAIN_C, FULL_C], capsize=5)
a1.set_ylabel("inserts/s (per-write durability)")
a1.set_title("Durability tax: throughput")
a1.grid(True, axis="y", ls=":", alpha=0.5)
p99 = [err(plain["perwrite_p99"]), err(full["perwrite_p99"])]
a2.bar(modes, [t[0] / 1000 for t in p99],
       yerr=[[t[1][0][0] / 1000 for t in p99], [t[1][1][0] / 1000 for t in p99]],
       color=[PLAIN_C, FULL_C], capsize=5)
a2.set_ylabel("p99 insert latency (ms)")
a2.set_title("Durability tax: tail latency")
a2.grid(True, axis="y", ls=":", alpha=0.5)
save(fig, "durability_tax")

# --- Fig B: group-commit batch-size sweep (the amortization curve) -------------
batches = [1, 10, 50, 200, 1000, 2000]
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for m, c, lab in ((full, FULL_C, "F_FULLFSYNC"), (plain, PLAIN_C, "Plain fsync")):
    ys, los, his = [], [], []
    for b in batches:
        med, lo, hi = m[f"gc_qps_b{b}"]
        ys.append(med); los.append(med - lo); his.append(hi - med)
    ax.errorbar(batches, ys, yerr=[los, his], marker="o", color=c, capsize=3, label=lab, lw=1.8)
ax.set_xscale("log")
ax.set_xlabel("batch size (inserts per fsync)")
ax.set_ylabel("inserts/s")
ax.set_title("Group commit: throughput vs. batch size")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(frameon=False)
# annotate the headline speedup under honest durability
sp = full["gc_qps_b2000"][0] / full["gc_qps_b1"][0]
ax.annotate(f"{sp:.1f}x at batch=2000\n(F_FULLFSYNC)", xy=(2000, full["gc_qps_b2000"][0]),
            xytext=(60, full["gc_qps_b2000"][0] * 0.6), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"))
save(fig, "groupcommit_sweep")

# --- Fig C: recovery sealed vs mutable vs N -----------------------------------
Ns = [1000, 3000, 6000]
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for key, c, lab, mk in (("sealed_ms", "#2ca02c", "Sealed (snapshot load)", "o"),
                        ("mutable_ms", "#d62728", "Mutable (WAL replay)", "s")):
    ys, los, his = [], [], []
    for n in Ns:
        med, lo, hi = full[f"{key}_n{n}"]
        ys.append(med); los.append(med - lo); his.append(hi - med)
    ax.errorbar(Ns, ys, yerr=[los, his], marker=mk, color=c, capsize=3, label=lab, lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("N (vectors)")
ax.set_ylabel("cold-open recovery time (ms)")
ax.set_title("Recovery: snapshot load vs. WAL replay")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(frameon=False)
save(fig, "recovery_sweep")

print("\nHeadline (d=128, median):")
print(f"  fsync floor : plain {plain['fsync_floor'][0]:.0f}/s  full {full['fsync_floor'][0]:.0f}/s "
      f"({plain['fsync_floor'][0]/full['fsync_floor'][0]:.0f}x)")
print(f"  per-write   : plain {plain['perwrite_qps'][0]:.0f}/s  full {full['perwrite_qps'][0]:.0f}/s")
print(f"  group commit: full b=2000 {full['gc_qps_b2000'][0]:.0f}/s  ({sp:.1f}x over b=1)")
print(f"  recovery N=6000: sealed {full['sealed_ms_n6000'][0]:.1f}ms  mutable {full['mutable_ms_n6000'][0]:.1f}ms "
      f"({full['mutable_ms_n6000'][0]/full['sealed_ms_n6000'][0]:.0f}x)")
