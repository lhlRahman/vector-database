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
HNSWLIB_COMMIT="d9b3608c83d83b46c96e25088cb1d729b29dcfe9"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_sha256() {
  file="$1"
  expected="$2"
  actual="$(sha256_file "$file")"
  if [ "$actual" != "$expected" ]; then
    echo "SHA256 mismatch for $file" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
}

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
  if [ "$DATASET" = sift ]; then
    verify_sha256 "$TARBALL" \
      "92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a"
  fi
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

if [ "$DATASET" = sift ]; then
  verify_sha256 "$DEST/sift_base.fvecs" \
    "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816"
  verify_sha256 "$DEST/sift_query.fvecs" \
    "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc"
  verify_sha256 "$DEST/sift_groundtruth.ivecs" \
    "2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f"
  verify_sha256 "$DEST/sift_learn.fvecs" \
    "331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea"
  echo "==> verified SIFT1M SHA256 manifest"
fi

# Vendor hnswlib (header-only) for the baseline comparison.
HNSW_DIR="$ROOT/third_party/hnswlib"
if [ ! -f "$HNSW_DIR/hnswlib/hnswlib.h" ]; then
  echo "==> cloning hnswlib into third_party/"
  mkdir -p "$ROOT/third_party"
  git clone https://github.com/nmslib/hnswlib "$HNSW_DIR"
  git -C "$HNSW_DIR" checkout --detach "$HNSWLIB_COMMIT"
else
  echo "==> hnswlib already vendored"
fi

actual_hnswlib_commit="$(git -C "$HNSW_DIR" rev-parse HEAD)"
if [ "$actual_hnswlib_commit" != "$HNSWLIB_COMMIT" ]; then
  echo "hnswlib revision mismatch" >&2
  echo "  expected: $HNSWLIB_COMMIT" >&2
  echo "  actual:   $actual_hnswlib_commit" >&2
  exit 1
fi
echo "==> verified hnswlib v0.9.0 ($HNSWLIB_COMMIT)"

echo "done. Next:"
echo "  make bench-ann     ANN_ARGS=\"--data datasets/$DATASET\""
echo "  make bench-hnswlib HNSWLIB_ARGS=\"--data datasets/$DATASET\""
