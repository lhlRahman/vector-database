#!/bin/bash
# Assemble a self-contained arXiv submission tarball: the .tex, the official
# ADMS/PVLDB acmart class, and the figure/data files it reads. No .bib is needed
# because the paper uses an inline thebibliography. Pass --verify to compile it.
#
#   scripts/make_arxiv.sh [--verify]
set -eu
cd "$(dirname "$0")/.."
SRC=docs/paper
OUT=build/arxiv
TECTONIC="$HOME/.local/bin/tectonic"
VERIFY=0
case "${1:-}" in
  "") ;;
  --verify) VERIFY=1 ;;
  *) echo "usage: scripts/make_arxiv.sh [--verify]" >&2; exit 2 ;;
esac

rm -rf "$OUT"; mkdir -p "$OUT/figs" "$OUT/data"
cp "$SRC/honest-durability.tex" "$SRC/acmart.cls" "$OUT/"
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

( cd "$OUT" && tar czf ../honest-durability-arxiv.tar.gz honest-durability.tex acmart.cls figs data )
echo "arXiv package -> build/honest-durability-arxiv.tar.gz"
ls -lh build/honest-durability-arxiv.tar.gz | awk '{print "  size:", $5}'

if [ "$VERIFY" -eq 1 ]; then
  [ -x "$TECTONIC" ] || {
    echo "ERROR: Tectonic not executable at $TECTONIC" >&2
    exit 1
  }
  echo "compile-checking the packaged source ..."
  ( cd "$OUT" && "$TECTONIC" -X compile honest-durability.tex >/dev/null 2>&1 \
      && echo "  compile OK: $(ls -l honest-durability.pdf | awk '{print $5}') bytes" \
      || { echo "  COMPILE FAILED"; exit 1; } )
fi
