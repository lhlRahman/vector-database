# ADMS / arXiv submission checklist

Build and compile the self-contained package from the repository root:

```sh
scripts/make_arxiv.sh --verify
```

The command creates `build/honest-durability-arxiv.tar.gz` with the manuscript,
the vendored official ADMS/PVLDB `acmart.cls`, and both canonical summary CSVs.
Missing assets or a missing Tectonic executable are fatal in verification mode.

## Must decide before submission

- [ ] Replace the anonymous author and affiliation for arXiv or a camera-ready
  version; retain anonymity if the active venue requires it.
- [ ] Set the artifact URL when a stable public repository/archive exists.
- [ ] Replace the PVLDB DOI, issue, and page placeholders when assigned.
- [ ] Select the arXiv license and categories (suggested primary `cs.DB`, with
  `cs.IR` or `cs.DC` as appropriate).

## Verified for round 4

- [x] Uses the ADMS 2026 call's linked PVLDB Volume 19 template:
  `acmart` commit `bc9fcacbdc559a577816812cb46c6d41d8bced2b`.
- [x] Tectonic produces an 8-page PDF with no undefined references or overfull
  boxes, below ADMS's 12 content-page limit.
- [x] The inline bibliography needs no external `.bib` or `.bbl` file.
- [x] The canonical bundle is run `20260720T044019Z-889559979de0` from clean
  relevant-source commit `889559979de0519a30de04e9aefbf67a3770e69f`.
- [x] The main evidence has 240 images, 2,880 recovery rows, and 14,040 operation
  rows; the paired sweep has 480 images and 30,720 write rows.
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
