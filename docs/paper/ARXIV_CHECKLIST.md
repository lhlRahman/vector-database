# arXiv / camera-ready checklist — "Honest Durability for Graph ANN"

Build the submission tarball: `scripts/make_arxiv.sh --verify`
-> `build/honest-durability-arxiv.tar.gz` (honest-durability.tex + 3 figure PDFs).

## Must decide (needs you)
- [ ] **Author block.** The .tex has `\author{Anonymous\\ \small \texttt{Under submission}}`.
  - For **arXiv**: arXiv is NOT anonymous — replace with real name(s) + affiliation + email.
  - For a **double-blind venue** submission: keep anonymous (do not upload the deanonymized version to arXiv before the anonymity deadline if the venue forbids it — check the CFP).
- [ ] **Public code URL.** The paper says "all code ... open" but prints no repository link.
  Add a URL (make the repo public first) to the Reproducibility section, or drop the "open" claim.
- [ ] **arXiv license.** Pick one at submission (default arXiv non-exclusive is fine; CC-BY if you want reuse).
- [ ] **arXiv categories.** Suggested: primary **cs.DB**; cross-list **cs.IR**, **cs.DC**.

## Verified (this session)
- [x] Compiles clean with tectonic (pdflatex-compatible packages only: geometry, amsmath,
      booktabs, graphicx, textcomp, microtype, hyperref). 0 overfull hboxes, 0 undefined refs.
- [x] Self-contained: standard `article` class, inline `thebibliography` (no external .bbl/.bib),
      figures are local PDFs under `figs/`.
- [x] All 16 citations web-verified; no `[unverified]` tags remain.
- [x] Every durability number is median-of-7 with min–max ranges; all cross-references resolve.
- [x] Passed two no-context 10-reviewer rounds (final: 10/10 >= weak-accept, 0 blocking issues).
- [x] Code: `make test` 50/50; ThreadSanitizer clean (0 races) on e2e; ASan+UBSan fuzz clean (WAL, sealed-segment).

## Before posting
- [ ] Re-run `scripts/make_arxiv.sh --verify` after any .tex edit.
- [ ] Fold in GIST1M results (see `GIST_INTEGRATION.md`) if the queued run finished.
- [ ] Sanity-check the PDF renders on a clean machine (arXiv uses its own TeX Live).
- [ ] Title/abstract for the arXiv web form match the paper's.

## Honest scope to keep in the abstract/threats (do NOT overclaim)
- Single macOS/APFS host; the ~155x fsync-floor gap is a platform artifact (smaller on Linux/NVMe).
- Durability numbers are synthetic-vector unless the real-SIFT run is folded in.
- Fork-and-crash validates post-fsync death, not true power loss (needs Linux dm-log-writes).
- Positioned as a measurement/methodology paper (DaMoN/ADMS scope), not a systems-novelty claim.
