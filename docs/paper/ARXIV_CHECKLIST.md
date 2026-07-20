# ADMS / arXiv submission checklist

Build and compile the self-contained package from the repository root:

```sh
scripts/make_arxiv.sh --verify
```

The command creates `build/honest-durability-arxiv.tar.gz` with the manuscript,
the vendored official ADMS/PVLDB `acmart.cls` and bibliography style, BibTeX and
PDF/A metadata, and both canonical summary CSVs. Missing assets, Tectonic, or
veraPDF are fatal in verification mode. Set `VERAPDF` and `JAVA_HOME` when the
validator is not on `PATH`.

## Must decide before submission

- [ ] Replace the anonymous author and affiliation for arXiv or a camera-ready
  version; retain anonymity if the active venue requires it.
- [ ] Set the artifact URL when a stable public repository/archive exists.
- [ ] Replace the PVLDB DOI, issue, and page placeholders when assigned.
- [ ] Select the arXiv license and categories (suggested primary `cs.DB`, with
  `cs.IR` or `cs.DC` as appropriate).

## Verified for round 5

- [x] Uses the ADMS 2026 call's linked PVLDB Volume 19 template:
  `acmart` commit `bc9fcacbdc559a577816812cb46c6d41d8bced2b`.
- [x] Tectonic produces an 8-page PDF with no undefined references or overfull
  boxes, below ADMS's 12 content-page limit.
- [x] All 24 citations resolve through packaged BibTeX and the official
  `ACM-Reference-Format.bst`; rendered references are alphabetically ordered.
- [x] The final two columns are balanced and veraPDF validates the output as
  PDF/A-2b with embedded fonts.
- [x] The canonical bundle is run `20260720T055159Z-9d93ba5fceab` from clean
  relevant-source commit `9d93ba5fceab945985c28e9f18e337bf2c7c1555`.
- [x] The main evidence has 240 images, 2,880 recovery rows, and 14,100 operation
  rows; the paired sweep has 480 images and 30,720 write rows.
- [x] All 30 strict-cap images lose exactly two exposed weak records and attain
  `Delta+ = L = M = .2` for one query without a cap overshoot.
- [x] Tests pass at 77/77 general, 30/30 end-to-end, 20/20 TCP, 24/24 committer,
  15/15 process-crash frontiers, and 661/661 WAL byte prefixes.
- [x] The exact published `.artifacts/recall-committer-evidence` recipe ran and
  the tracked canonical bundle revalidates byte-for-byte.
- [x] The physical `dm-log-writes` experiment was not run and is not claimed.

## Before posting

- [ ] Re-run `scripts/make_arxiv.sh --verify` after any manuscript edit.
- [ ] Verify a clean `git archive` compile and inspect the final PDF visually.
- [ ] Ensure the title and abstract in the submission form match the PDF.

## Scope that must remain explicit

- The new committer timings are from one Apple/APFS host and a small synthetic
  workload with one query set and two fixed HNSW graphs.
- The legacy tax/recovery study spans Apple and AWS Linux but lacks full modern
  provenance parity; it is labeled as pre-committer evidence.
- Process death, logical image recovery, and byte cuts are not physical power
  loss. The destructive Linux block-replay run remains pending.
- The implementation is single-node and insert-only. The root lock assumes
  cooperating processes and local filesystem `flock` semantics.
