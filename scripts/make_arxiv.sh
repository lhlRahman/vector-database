#!/bin/bash
# Assemble a self-contained arXiv/ADMS submission tarball: TeX, BibTeX and
# PDF/A metadata, the official PVLDB acmart class, and referenced data/assets.
# Pass --verify to compile and enforce the submission-format checks.
#
#   scripts/make_arxiv.sh [--verify]
set -eu
cd "$(dirname "$0")/.."
SRC=docs/paper
OUT=build/arxiv
TECTONIC="$HOME/.local/bin/tectonic"
VERAPDF=${VERAPDF:-$(command -v verapdf 2>/dev/null || true)}
VERIFY=0
case "${1:-}" in
  "") ;;
  --verify) VERIFY=1 ;;
  *) echo "usage: scripts/make_arxiv.sh [--verify]" >&2; exit 2 ;;
esac

rm -rf "$OUT"
rm -f build/honest-durability-arxiv.tar.gz
mkdir -p "$OUT/figs" "$OUT/data"
cp "$SRC/honest-durability.tex" "$SRC/honest-durability.bib" \
  "$SRC/honest-durability.xmpdata" "$SRC/acmart.cls" \
  "$SRC/ACM-Reference-Format.bst" "$OUT/"
missing=0

# Copy only the figures actually referenced by \includegraphics.
figs=$(grep -oE 'figs/[A-Za-z0-9_]+\.pdf' "$SRC/honest-durability.tex" | sort -u)
for f in $figs; do
  if [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$OUT/$f"
  else
    echo "ERROR: missing $SRC/$f" >&2
    missing=1
  fi
done
echo "Referenced figures:"; echo "$figs" | sed 's/^/  /'

# PGFPlots reads these files at TeX compile time, so they must accompany source.
data_files=$(grep -oE 'data/[A-Za-z0-9_./-]+\.csv' "$SRC/honest-durability.tex" | sort -u)
for f in $data_files; do
  if [ -f "$SRC/$f" ]; then
    mkdir -p "$OUT/$(dirname "$f")"
    cp -L "$SRC/$f" "$OUT/$f"
  else
    echo "ERROR: missing $SRC/$f" >&2
    missing=1
  fi
done
echo "Referenced data:"; echo "$data_files" | sed 's/^/  /'
[ "$missing" -eq 0 ] || exit 1

# Include the executed device-level power-loss evidence and its tracked digest.
cp "$SRC/data/powerloss_committer_sha256.txt" "$OUT/data/"
cp -R "$SRC/data/powerloss_committer" "$OUT/data/"
if command -v sha256sum >/dev/null 2>&1; then
  ( cd "$OUT/data" && sha256sum -c powerloss_committer_sha256.txt >/dev/null )
elif command -v shasum >/dev/null 2>&1; then
  ( cd "$OUT/data" && shasum -a 256 -c powerloss_committer_sha256.txt >/dev/null )
else
  echo "ERROR: sha256sum or shasum is required to verify packaged evidence" >&2
  exit 1
fi
echo "Physical replay evidence: data/powerloss_committer (SHA-256 verified)"

if [ "$VERIFY" -eq 1 ]; then
  [ -x "$TECTONIC" ] || {
    echo "ERROR: Tectonic not executable at $TECTONIC" >&2
    exit 1
  }
  [ -n "$VERAPDF" ] && [ -x "$VERAPDF" ] || {
    echo "ERROR: veraPDF is required for --verify (set VERAPDF or install verapdf)" >&2
    exit 1
  }
  echo "compile-checking the packaged source ..."
  ( cd "$OUT"
    "$TECTONIC" -X compile --keep-logs honest-durability.tex >tectonic.out 2>&1 || {
      cat tectonic.out >&2
      echo "  COMPILE FAILED" >&2
      exit 1
    }
    # Tectonic records every pass in stdout; layout warnings before BibTeX are
    # transient. The retained .log is the final TeX pass and is authoritative.
    if grep -Eq 'Citation .* undefined|There were undefined (citations|references)|Overfull \\[hv]box|Package balance Warning|Class acmart Warning: \\vspace should only be used' \
        honest-durability.log; then
      grep -E 'Citation .* undefined|There were undefined (citations|references)|Overfull \\[hv]box|Package balance Warning|Class acmart Warning: \\vspace should only be used' \
        honest-durability.log >&2 || true
      echo "  FORMAT CHECK FAILED" >&2
      exit 1
    fi
    if [ -f honest-durability.blg ] && grep -Eq '^Warning--|I couldn.t open database' honest-durability.blg; then
      grep -E '^Warning--|I couldn.t open database' honest-durability.blg >&2 || true
      echo "  BIBLIOGRAPHY CHECK FAILED" >&2
      exit 1
    fi
    pdfa_result=$("$VERAPDF" --format text --flavour 2b honest-durability.pdf 2>/dev/null) || {
      printf '%s\n' "$pdfa_result" >&2
      echo "  PDF/A VALIDATION FAILED" >&2
      exit 1
    }
    printf '%s\n' "$pdfa_result" | grep -Eq '^PASS .* 2b$' || {
      printf '%s\n' "$pdfa_result" >&2
      echo "  PDF/A-2b PASS MARKER MISSING" >&2
      exit 1
    }
    echo "  compile OK: $(ls -l honest-durability.pdf | awk '{print $5}') bytes"
    printf '  %s\n' "$pdfa_result"
  )
fi

( cd "$OUT" && tar czf ../honest-durability-arxiv.tar.gz \
    honest-durability.tex honest-durability.bib honest-durability.xmpdata \
    acmart.cls ACM-Reference-Format.bst figs data )
echo "arXiv package -> build/honest-durability-arxiv.tar.gz"
ls -lh build/honest-durability-arxiv.tar.gz | awk '{print "  size:", $5}'
