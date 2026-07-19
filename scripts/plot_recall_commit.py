#!/usr/bin/env python3
"""Plot the recall-aware committer's SLA: measured crash ΔRecall vs target ε."""
import csv, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV=sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT,"docs/paper/data/recall_commit_sift.csv")
OUT=os.path.join(ROOT,"docs/paper/figs"); os.makedirs(OUT,exist_ok=True)
rows=list(csv.DictReader(open(CSV)))
eps=sorted({float(r["eps"]) for r in rows})
def series(rg): return [next(float(r["mean_dRecall"]) for r in rows if r["regime"]==rg and float(r["eps"])==e) for e in eps]
fig,ax=plt.subplots(figsize=(5.4,3.8))
ax.plot(eps,eps,"k:",lw=1.3,label=r"SLA target ($\Delta$Recall $=\varepsilon$)")
ax.plot(eps,series("benign"),"o-",color="#1f77b4",lw=1.8,ms=6,label="benign (measured)")
ax.plot(eps,series("adversarial"),"^--",color="#ff7f0e",lw=1.8,ms=6,label="adversarial (measured)")
ax.set_xlabel(r"target recall-staleness budget $\varepsilon$"); ax.set_ylabel(r"measured crash $\Delta$recall@10 (mean)")
ax.set_title("Recall-aware committer: crash loss vs. SLA (SIFT1M)")
ax.grid(True,ls=":",alpha=0.5); ax.legend(frameon=False,fontsize=8,loc="upper left")
ax.fill_between(eps,eps,max(series("adversarial"))*1.05,color="0.93",zorder=0)
for ext in ("pdf","png"): fig.savefig(os.path.join(OUT,f"recall_commit_sla.{ext}"),bbox_inches="tight",dpi=150)
print("wrote recall_commit_sla.pdf/.png")
