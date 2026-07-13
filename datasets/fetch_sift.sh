#!/usr/bin/env bash
# Fetch a standard ANN benchmark dataset (TEXMEX) + vendor hnswlib for the
# baseline. Requires network (blocked in some sandboxes — run on a real host).
#
#   datasets/fetch_sift.sh          # SIFT1M (128d, 1M vectors, ships ground truth)
#   datasets/fetch_sift.sh gist     # GIST1M (960d, 1M vectors) — high-dim stress case
#
# After it runs:
#   make bench-ann      ANN_ARGS="--data datasets/sift"
#   make bench-hnswlib  HNSWLIB_ARGS="--data datasets/sift"   # baseline
set -euo pipefail

DATASET="${1:-sift}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

case "$DATASET" in
  sift) URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" ;;
  gist) URL="ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" ;;
  siftsmall) URL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" ;;
  *) echo "unknown dataset '$DATASET' (use: sift | gist | siftsmall)"; exit 1 ;;
esac

TARBALL="$HERE/${DATASET}.tar.gz"
DEST="$HERE/${DATASET}"

if [ ! -d "$DEST" ]; then
  echo "==> downloading $DATASET from $URL"
  curl -fSL --retry 3 -o "$TARBALL" "$URL"
  echo "==> extracting"
  tar -xzf "$TARBALL" -C "$HERE"
  # tarball extracts to <dataset>/ (siftsmall extracts to siftsmall/); the files
  # inside are named <dataset>_base.fvecs etc. bench_ann expects sift_base.fvecs;
  # for siftsmall the files are siftsmall_*.fvecs — bench_ann's --data expects the
  # sift_ prefix, so symlink/rename if needed.
  echo "==> $DATASET ready in $DEST"
  ls -la "$DEST"
else
  echo "==> $DATASET already present in $DEST"
fi

# Vendor hnswlib (header-only) for the baseline comparison.
HNSW_DIR="$ROOT/third_party/hnswlib"
if [ ! -f "$HNSW_DIR/hnswlib/hnswlib.h" ]; then
  echo "==> cloning hnswlib into third_party/"
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/nmslib/hnswlib "$HNSW_DIR"
else
  echo "==> hnswlib already vendored"
fi

echo "done. Next:"
echo "  make bench-ann     ANN_ARGS=\"--data datasets/$DATASET\""
echo "  make bench-hnswlib HNSWLIB_ARGS=\"--data datasets/$DATASET\""
