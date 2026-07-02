#!/usr/bin/env bash
# 03_build_dataset.sh — build a HYPIR training parquet from a folder of images.
#
# HYPIR's RealESRGANDataset (crop_type=none, out_size=512) requires every GT
# image to be EXACTLY out_size×out_size (512×512). So either:
#   * pass CROP=1 (default) to slice each image into 512×512 patches first, or
#   * pass CROP=0 and ensure your images are already 512×512 (or set
#     CROP_TYPE=random/center in 04_train.sh).
#
# Output: a parquet (columns: image_path[absolute], prompt) consumed by
# configs/sd2_train.yaml's file_meta. No official code is modified.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

DATA_DIR="${DATA_DIR:-}"
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: set DATA_DIR to a folder of (high-quality) images." >&2
    echo "       e.g. DATA_DIR=/data/LSDIR bash $0" >&2
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2; exit 1
fi

PARQUET_OUT="${PARQUET_OUT:-$DATA_DIR/hypir_train.parquet}"
# Make it absolute.
PARQUET_OUT="$(mkdir -p "$(dirname "$PARQUET_OUT")" && cd "$(dirname "$PARQUET_OUT")" && pwd)/$(basename "$PARQUET_OUT")"

# Build-dataset only needs polars + pillow; install on demand if missing.
if ! python -c "import polars, PIL" 2>/dev/null; then
    echo "--- installing polars pillow ---"
    pip install --trusted-host pypi.org --trusted-host pypi.python.org \
        --trusted-host files.pythonhosted.org --timeout 600 --retries 10 polars pillow
fi

export DATA_DIR PARQUET_OUT
python "$SCRIPT_DIR/build_dataset.py"

echo "=== [03] Done. Parquet: $PARQUET_OUT ==="
echo "    Next: PARQUET_PATH=$PARQUET_OUT bash $SCRIPT_DIR/04_train.sh"
