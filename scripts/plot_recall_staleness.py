#!/usr/bin/env python3
"""Plot recall-at-risk vs. the un-durable window W (the recall-bounded durability
result). Empirical (~W/N, benign) vs. the distribution-free worst-case min(W,k)/k.

    /usr/bin/python3 scripts/plot_recall_staleness.py [csv] [out_prefix] [N]
"""
import csv as _csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs/paper/data/recall_staleness_sift1m.csv")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "recall_staleness_sift1m"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
OUT = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(OUT, exist_ok=True)

W, emp, adv, wc = [], [], [], []
with open(CSV) as f:
    for r in _csv.DictReader(f):
        W.append(int(r["W"]))
        emp.append(float(r["recall_at_risk_empirical"]))
        adv.append(float(r.get("recall_at_risk_adversarial", "0") or "0"))
        wc.append(float(r["worst_case_bound_min_Wk_over_k"]))

RESOLUTION = 1e-4  # 1000 queries x k=10 -> smallest measurable nonzero recall-at-risk
fig, ax = plt.subplots(figsize=(5.6, 3.8))
# Worst-case bound (provable, distribution-free).
ax.plot(W, wc, "s--", color="#d62728", lw=1.8, ms=4, label=r"worst case $\min(W,k)/k$ (provable)")
# Adversarial: recent inserts are the query-hot vectors.
mAW = [w for w, a in zip(W, adv) if a > 0]
mA = [a for a in adv if a > 0]
if mA:
    ax.plot(mAW, mA, "^-", color="#ff7f0e", lw=1.6, ms=5, label="adversarial (query-hot inserts)")
# W/N reference (benign expectation).
ax.plot(W, [w / N for w in W], ":", color="gray", lw=1.4, label=r"$W/N$ (benign expectation)")
# Empirical measured points (only where above measurement resolution).
mW = [w for w, e in zip(W, emp) if e > 0]
mE = [e for e in emp if e > 0]
ax.plot(mW, mE, "o-", color="#1f77b4", lw=1.8, ms=5, label="measured (SIFT1M, benign)")
ax.axhspan(0, RESOLUTION, color="0.9", zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("un-durable window $W$ (un-synced inserts)")
ax.set_ylabel("recall@10 at risk from a crash")
ax.set_title("Recall-bounded durability (SIFT1M, $N{=}10^6$)")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(frameon=False, fontsize=8, loc="lower right")
# Annotate the operating point: batch = 2000.
if 2000 in W:
    e2000 = emp[W.index(2000)]
    ax.annotate(f"batch=2000: {e2000*100:.2f}% risk\nvs 100% worst case",
                xy=(2000, max(e2000, RESOLUTION)), xytext=(30, 0.2), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"{PREFIX}.{ext}"), bbox_inches="tight", dpi=150)
print(f"wrote {PREFIX}.pdf / .png")

# Console summary at the group-commit operating points.
print("\nrecall-at-risk (empirical) vs worst-case, key W:")
for w in (1000, 2000, 5000):
    if w in W:
        i = W.index(w)
        print(f"  W={w:>5}: empirical {emp[i]*100:.3f}%   worst-case {wc[i]*100:.0f}%   (W/N={w/N*100:.3f}%)")
