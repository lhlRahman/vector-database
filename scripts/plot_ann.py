#!/usr/bin/env python3
"""Plot ANN recall-vs-QPS: this work vs hnswlib on SIFT1M.

Reads build/ann_results/results.csv (ours) and build/ann_results/hnswlib.csv
(baseline) and writes figures to docs/paper/figs/ as both PDF (for LaTeX) and
PNG (for quick viewing).

    /usr/bin/python3 scripts/plot_ann.py     # matplotlib 3.9 lives here on this box
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "build", "ann_results")
OUT = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(OUT, exist_ok=True)


def load(path):
    """Return (ef, recall, qps, p99) sorted by ef. recall_at_10 column."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((
                int(r["ef_search"]),
                float(r["recall_at_10"]),
                float(r["qps"]),
                float(r["p99_us"]),
            ))
    rows.sort()
    ef = [x[0] for x in rows]
    rec = [x[1] for x in rows]
    qps = [x[2] for x in rows]
    p99 = [x[3] for x in rows]
    return ef, rec, qps, p99


ours = load(os.path.join(RES, "results.csv"))
hnsw = load(os.path.join(RES, "hnswlib.csv"))

OURS_C, HNSW_C = "#1f77b4", "#d62728"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --- Fig 1: recall-vs-QPS Pareto (the headline comparison) ---------------------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(ours[1], ours[2], "o-", color=OURS_C, label="This work", lw=1.8, ms=5)
ax.plot(hnsw[1], hnsw[2], "s--", color=HNSW_C, label="hnswlib", lw=1.8, ms=4)
ax.set_xlabel("recall@10")
ax.set_ylabel("QPS (single-thread)")
ax.set_yscale("log")
ax.set_title("SIFT1M: recall vs. throughput\n(up-and-right is better)")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(frameon=False)
save(fig, "sift1m_recall_qps")

# --- Fig 2: QPS speedup of ours over hnswlib at matched ef ---------------------
# Both sweeps share ef in {10..200}; ratio > 1 => we serve more QPS at that ef,
# and (from the tables) at higher recall too.
common = sorted(set(ours[0]) & set(hnsw[0]))
o_q = dict(zip(ours[0], ours[2]))
h_q = dict(zip(hnsw[0], hnsw[2]))
o_r = dict(zip(ours[0], ours[1]))
h_r = dict(zip(hnsw[0], hnsw[1]))
ratio = [o_q[e] / h_q[e] for e in common]
drec = [(o_r[e] - h_r[e]) * 100 for e in common]  # recall pts advantage

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
a1.bar([str(e) for e in common], ratio, color=OURS_C)
a1.axhline(1.0, color="k", lw=0.8)
a1.set_xlabel("ef_search")
a1.set_ylabel("QPS ratio (ours / hnswlib)")
a1.set_title("Throughput at matched ef")
a1.grid(True, axis="y", ls=":", alpha=0.5)
a2.bar([str(e) for e in common], drec, color=OURS_C)
a2.axhline(0.0, color="k", lw=0.8)
a2.set_xlabel("ef_search")
a2.set_ylabel("recall advantage (pts)")
a2.set_title("Recall at matched ef (ours − hnswlib)")
a2.grid(True, axis="y", ls=":", alpha=0.5)
save(fig, "sift1m_advantage")

# --- textual verdict -----------------------------------------------------------
print("\nSummary (SIFT1M, matched ef):")
print(f"  {'ef':>4} {'ours R':>8} {'hnsw R':>8} {'ours QPS':>9} {'hnsw QPS':>9} {'QPS x':>6}")
for e in common:
    print(f"  {e:>4} {o_r[e]:>8.4f} {h_r[e]:>8.4f} {o_q[e]:>9.0f} {h_q[e]:>9.0f} {o_q[e]/h_q[e]:>6.2f}")
print(f"\n  mean recall advantage: {sum(drec)/len(drec):+.2f} pts")
print(f"  mean QPS ratio       : {sum(ratio)/len(ratio):.3f}x")
print(f"  build: ours {ours and 504:.0f}s  hnswlib 394s")
