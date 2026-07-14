# GIST1M integration plan (apply once the queued runs finish)

Runs producing the numbers (auto-run by `scripts/queue_after_gist.sh` after GIST-ours):
- ours:     `build/gist1m_run.log`      -> `build/ann_results/results.csv`
- hnswlib:  `build/gist1m_hnswlib.log`  -> `build/ann_results/hnswlib.csv`
- real-SIFT durability: `build/durability_realsift_{plain,full}.log` -> `durability_{plain,full}_real_d128.csv`

## Steps
1. Archive CSVs (results.csv is overwritten by the GIST run; copy before it's clobbered again):
   - `cp build/ann_results/results.csv  docs/paper/data/ann_gist1m_ours.csv`
   - `cp build/ann_results/hnswlib.csv  docs/paper/data/ann_gist1m_hnswlib.csv`
   - `cp build/durability_results/durability_{plain,full}_real_d128.csv docs/paper/data/`
2. Figures:
   - `/usr/bin/python3 scripts/plot_ann.py docs/paper/data/ann_gist1m_ours.csv docs/paper/data/ann_gist1m_hnswlib.csv gist1m GIST1M`
     -> writes `docs/paper/figs/gist1m_recall_qps.{pdf,png}`.

## .tex edits (competitiveness section, §7)
- **Abstract**: "reaches 0.999 recall@10 on SIFT1M" -> "on SIFT1M and GIST1M (960-d)".
- **§7 prose**: add one paragraph after the SIFT discussion:
  "On the higher-dimensional GIST1M (960-d, 1M vectors) the same picture holds:
   recall@10 climbs to <X> and the two indexes remain Pareto-equivalent (ours
   <r>/<qps> vs. hnswlib <r>/<qps> at ef=<e>); build time <t>s vs. <t_h>s."
- **Table**: add a compact GIST1M table (same 5-col ours-vs-hnswlib layout as tab:recall)
  OR extend tab:recall with a GIST block. Fill from the CSVs (ef, recall, qps x2).
- **Optional**: include `figs/gist1m_recall_qps.pdf` as a second Pareto panel.
- **Threats**: remove "(i) a GIST1M (960-d) run" from remaining work (now done).

## Real-SIFT durability (robustness confirmation, §4)
- Compare `durability_*_real_d128.csv` to the synthetic d=128 numbers. Expected:
  within run-to-run spread (tax is flush-bound, not data-dependent). Add one line to
  §4's dimension-robustness paragraph: "Re-running the tax on real SIFT1M embeddings
  (vs. synthetic) gives <floor>/<perwrite>/<speedup>, within the trial spread — the
  tax does not depend on the vector distribution." Then drop the synthetic-vector
  caveat from Threats.

## Recompile + re-verify
- `tectonic -X compile honest-durability.tex` (expect 0 overfull, 0 undefined).
- Re-run one review round to confirm no new inconsistencies.
