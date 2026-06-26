#!/usr/bin/env bash
# 02_run_inference.sh — run the official run_example.py on a single sample image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

TRIPOSPLAT_DIR="${TRIPOSPLAT_DIR:-$REPO_DIR/../TripoSplat}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRIPOSPLAT_DIR/output}"

echo "=== [02] Running TripoSplat inference ==="
echo "  code dir: $TRIPOSPLAT_DIR"
echo "  output:   $OUTPUT_DIR"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "  GPU:      physical $CUDA_VISIBLE_DEVICES (cuda:0 in-process)  [GPU=N to change]"
else
    echo "  GPU:      default cuda:0 (= first visible)  [set GPU=N to pin a card]"
fi

if [ ! -d "$TRIPOSPLAT_DIR" ]; then
    echo "ERROR: TripoSplat code dir not found at $TRIPOSPLAT_DIR." >&2
    exit 1
fi
if [ ! -e "$TRIPOSPLAT_DIR/ckpts" ]; then
    echo "ERROR: $TRIPOSPLAT_DIR/ckpts missing. Run 01_download_models.sh first." >&2
    exit 1
fi

# run_example.py loads weights from ./ckpts/... (a symlink to the shared
# model store) and reads ./static/example_inputs/... — launch from repo root.
cd "$TRIPOSPLAT_DIR"
python run_example.py

# run_example.py writes outputs to CWD (repo root). Move them into OUTPUT_DIR
# so they're collected in one place. (Official code is not modified.)
mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
mv -f output.ply output.splat preprocessed_image.webp "$OUTPUT_DIR/" 2>/dev/null || true
mv -f output_*.ply "$OUTPUT_DIR/" 2>/dev/null || true
shopt -u nullglob

echo "--- outputs in $OUTPUT_DIR ---"
ls -lh "$OUTPUT_DIR"/ 2>/dev/null || true

echo "=== [02] Done. Outputs in: $OUTPUT_DIR ==="
